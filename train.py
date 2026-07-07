import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

import Parameters as P
from LossFunctionWithHelpers import ContrastiveSoftDTW
from embeddingModel import EmbeddingModel
from newDataLoader import build_dataloaders
from textEmbedding import TextEmbedding


USE_AMP = torch.cuda.is_available() and os.environ.get("USE_AMP", "1") == "1"
AMP_DTYPE = torch.float16


def compute_stride(window_size, stride_ratio, window_overlap_mode):
    if window_overlap_mode == "no_overlap":
        return window_size
    if window_overlap_mode == "light_overlap":
        return max(1, window_size // 2)
    if window_overlap_mode == "dense_overlap":
        return max(1, window_size // 4)
    if window_overlap_mode == "custom":
        return max(1, int(window_size * stride_ratio))
    raise ValueError(f"Unknown window_overlap_mode: {window_overlap_mode!r}")


def weights_dir(job_id):
    path = os.path.join(os.path.dirname(__file__), "Weights", job_id)
    os.makedirs(path, exist_ok=True)
    return path


def model_config(stride):
    return {
        "window_size": P.window_size,
        "stride": stride,
        "stride_ratio": P.stride_ratio,
        "window_overlap_mode": P.window_overlap_mode,
        "vector_size": P.vector_size,
        "lang": P.lang,
        "negative_mode": P.negative_mode,
        "num_negatives": P.num_negatives,
        "use_bilstm": P.use_bilstm,
        "bilstm_layers": P.bilstm_layers,
        "bilstm_hidden_dim": P.bilstm_hidden_dim,
        "contrastive_soft_dtw_gamma": P.contrastive_soft_dtw_gamma,
        "contrastive_margin": P.contrastive_margin,
        "contrastive_temperature": P.contrastive_temperature,
    }


def save_model_weights(model, text_embedder, job_id, config):
    path = os.path.join(weights_dir(job_id), "model_latest.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_model_state_dict": model.state_dict(),
            "text_embedder_state_dict": text_embedder.state_dict(),
            "text_embedding_class": "TextEmbedding",
            "model_config": config,
        },
        path,
    )
    return path


def save_checkpoint(model, text_embedder, optimizer, scheduler, scaler, epoch, job_id, config):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "image_model_state_dict": model.state_dict(),
            "text_embedder_state_dict": text_embedder.state_dict(),
            "text_embedding_class": "TextEmbedding",
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "model_config": config,
        },
        os.path.join(weights_dir(job_id), "checkpoint_latest.pth"),
    )


def extract_model_state(loaded):
    if isinstance(loaded, dict) and "image_model_state_dict" in loaded:
        return loaded["image_model_state_dict"]
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    return loaded


def build_text_embedding():
    # TextEmbedding is frozen; only the image encoder is optimized.
    text_embedder = TextEmbedding(embedding_dim=P.vector_size)
    text_embedder = text_embedder.to(P.device)
    for parameter in text_embedder.parameters():
        parameter.requires_grad_(False)
    text_embedder.eval()
    return text_embedder


def build_image_embedding(stride):
    return EmbeddingModel(
        window_size=P.window_size,
        stride=stride,
        vector_size=P.vector_size,
        device=P.device,
        use_flip=(P.lang.lower() == "arabic"),
        use_bilstm=P.use_bilstm,
        bilstm_layers=P.bilstm_layers,
        bilstm_hidden_dim=P.bilstm_hidden_dim,
    )


def embed_single_text(text_embedder, text):
    with torch.no_grad():
        embedding = text_embedder(text)
    return F.normalize(embedding.float(), p=2, dim=-1)


def compute_similarity_lists(text_embedder, norm_img, pos_texts, neg_texts):
    sim_pos_list = []
    sim_neg_list = []

    for sample_idx, pos_text in enumerate(pos_texts):
        norm_pos_text = embed_single_text(text_embedder, pos_text)
        sim_pos = torch.einsum("sv,tv->st", norm_pos_text, norm_img[sample_idx])
        sim_pos_list.append(sim_pos)

        sample_neg_sims = []
        for neg_text in neg_texts[sample_idx]:
            norm_neg_text = embed_single_text(text_embedder, neg_text)
            sample_neg_sims.append(
                torch.einsum("tv,sv->ts", norm_neg_text, norm_img[sample_idx])
            )
        sim_neg_list.append(sample_neg_sims)

    return sim_pos_list, sim_neg_list


def compute_batch_loss(image_embedder, text_embedder, criterion, images, pos_texts, neg_texts):
    # Image encoder is the only trainable branch.
    with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        img_emb = image_embedder(images)
    norm_img = F.normalize(img_emb.float(), p=2, dim=-1)

    # Frozen character text embeddings feed direct cosine similarities.
    sim_pos_list, sim_neg_list = compute_similarity_lists(
        text_embedder, norm_img, pos_texts, neg_texts
    )
    return criterion.forward_varlen(sim_pos_list, sim_neg_list)


def train_one_epoch(model, text_embedder, criterion, optimizer, scaler, loader):
    model.train()
    text_embedder.eval()
    total = 0.0

    for batch_idx, (images, pos_texts, neg_texts) in enumerate(loader):
        images = images.to(P.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        loss, stats = compute_batch_loss(
            model, text_embedder, criterion, images, pos_texts, neg_texts
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        print(
            f"batch={batch_idx + 1}/{len(loader)} "
            f"loss={loss.item():.4f} pos={stats['cost_pos']:.2f} neg={stats['cost_neg']:.2f}",
            flush=True,
        )

    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, text_embedder, criterion, loader):
    model.eval()
    text_embedder.eval()
    total = 0.0

    for images, pos_texts, neg_texts in loader:
        images = images.to(P.device, non_blocking=True)
        loss, _ = compute_batch_loss(model, text_embedder, criterion, images, pos_texts, neg_texts)
        total += loss.item()

    return total / max(len(loader), 1)


def train(model, text_embedder, criterion, train_loader, valid_loader, args, config):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=P.device)
        model.load_state_dict(extract_model_state(checkpoint))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1

    history = []
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        print(f"epoch={epoch + 1}/{args.epochs}", flush=True)

        train_loss = train_one_epoch(model, text_embedder, criterion, optimizer, scaler, train_loader)
        val_loss = validate(model, text_embedder, criterion, valid_loader)
        scheduler.step()

        save_checkpoint(model, text_embedder, optimizer, scheduler, scaler, epoch, args.job_id, config)
        save_model_weights(model, text_embedder, args.job_id, config)

        history.append(train_loss)
        elapsed = time.time() - started
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train the alignment model")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--pretrained_weights", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride_ratio", type=float, default=None)
    parser.add_argument(
        "--window_overlap_mode",
        choices=["no_overlap", "light_overlap", "dense_overlap", "custom"],
        default=None,
    )
    parser.add_argument("--negative_mode", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_negatives", type=int, default=None)
    parser.add_argument("--use_bilstm", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def apply_overrides(args):
    if args.window_size is not None:
        P.window_size = args.window_size
    if args.stride_ratio is not None:
        P.stride_ratio = args.stride_ratio
    if args.window_overlap_mode is not None:
        P.window_overlap_mode = args.window_overlap_mode
    if args.negative_mode is not None:
        P.negative_mode = args.negative_mode.lower()
    if args.num_negatives is not None:
        P.num_negatives = args.num_negatives
        import newDataLoader

        newDataLoader.num_negatives = args.num_negatives
    if args.use_bilstm is not None:
        P.use_bilstm = args.use_bilstm

    if args.finetune:
        args.learning_rate = P.finetune_learning_rate if args.learning_rate is None else args.learning_rate
        args.epochs = P.finetune_epochs if args.epochs is None else args.epochs
    else:
        args.learning_rate = P.learning_rate if args.learning_rate is None else args.learning_rate
        args.epochs = P.epochs if args.epochs is None else args.epochs


def select_dataloaders(args):
    data_dir = args.data_dir
    if data_dir is None and args.finetune:
        data_dir = P.finetune_data_dir
    return build_dataloaders(data_dir)


def main():
    args = parse_args()
    apply_overrides(args)

    if args.resume and args.pretrained_weights:
        raise SystemExit("Use either --resume or --pretrained_weights, not both.")

    stride = compute_stride(P.window_size, P.stride_ratio, P.window_overlap_mode)
    train_loader, valid_loader, _test_loader = select_dataloaders(args)

    text_embedder = build_text_embedding()
    model = build_image_embedding(stride).to(P.device)

    if args.pretrained_weights:
        loaded = torch.load(args.pretrained_weights, map_location=P.device)
        model.load_state_dict(extract_model_state(loaded))

    criterion = ContrastiveSoftDTW(
        gamma=P.contrastive_soft_dtw_gamma,
        use_cuda=torch.cuda.is_available(),
        margin=P.contrastive_margin,
        temperature=P.contrastive_temperature,
    )
    config = model_config(stride)

    print(
        f"job_id={args.job_id} device={P.device} epochs={args.epochs} lr={args.learning_rate} "
        f"window_size={P.window_size} stride={stride} negatives={P.num_negatives}",
        flush=True,
    )
    print(
        f"text_embedder=TextEmbedding use_bilstm={P.use_bilstm} "
        f"soft_dtw_gamma={P.contrastive_soft_dtw_gamma}",
        flush=True,
    )

    train(model, text_embedder, criterion, train_loader, valid_loader, args, config)


if __name__ == "__main__":
    main()
