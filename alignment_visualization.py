import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def hard_restricted_dtw_path(sim, temperature=0.07):
    centered = sim - sim.mean(dim=0, keepdim=True)
    dist = -torch.log_softmax(centered / temperature, dim=0)
    dist = dist.detach().cpu()

    text_steps, image_steps = dist.shape
    dp = torch.full((text_steps, image_steps), float("inf"))
    back = [[None for _ in range(image_steps)] for _ in range(text_steps)]
    dp[0, 0] = dist[0, 0]

    for j in range(1, image_steps):
        dp[0, j] = dp[0, j - 1] + dist[0, j]
        back[0][j] = (0, j - 1)

    for i in range(1, text_steps):
        for j in range(1, image_steps):
            diag = dp[i - 1, j - 1]
            stay = dp[i, j - 1]
            if diag <= stay:
                dp[i, j] = diag + dist[i, j]
                back[i][j] = (i - 1, j - 1)
            else:
                dp[i, j] = stay + dist[i, j]
                back[i][j] = (i, j - 1)

    i, j = text_steps - 1, image_steps - 1
    path = [(i, j)]
    while back[i][j] is not None:
        i, j = back[i][j]
        path.append((i, j))
    path.reverse()
    return path


def _text_tokens(text_encoder, text):
    if hasattr(text_encoder, "tokenize_visual_units"):
        return text_encoder.tokenize_visual_units(text)
    return list(text)


@torch.no_grad()
def save_d3tw_visualization(
    model,
    text_encoder,
    valid_loader,
    criterion,
    epoch,
    job_id,
    device,
):
    del criterion
    try:
        images, pos_texts, _neg_texts = next(iter(valid_loader))
    except StopIteration:
        return None

    image = images[:1].to(device, non_blocking=True)
    pos_text = pos_texts[0]

    model.eval()
    text_encoder.eval()

    img_emb = model(image)
    norm_img = F.normalize(img_emb[0].float(), p=2, dim=-1)
    norm_text = F.normalize(text_encoder(pos_text).float(), p=2, dim=-1)
    if norm_text.numel() == 0 or norm_img.numel() == 0:
        return None

    sim = torch.einsum("tv,sv->ts", norm_text, norm_img)
    path = hard_restricted_dtw_path(sim)
    path_y = [point[0] for point in path]
    path_x = [point[1] for point in path]

    output_dir = os.path.join(os.path.dirname(__file__), "Weights", job_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"d3tw_epoch_{epoch:04d}.png")

    tokens = _text_tokens(text_encoder, pos_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(sim.detach().cpu().numpy(), aspect="auto", origin="upper")
    ax.plot(path_x, path_y, color="red", linewidth=1.5)
    ax.set_xlabel("image windows")
    ax.set_ylabel("text tokens")
    if len(tokens) <= 40:
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)
    title_text = pos_text.strip()
    if len(title_text) > 80:
        title_text = title_text[:77] + "..."
    ax.set_title(f"Epoch {epoch} D3TW: {title_text}")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
