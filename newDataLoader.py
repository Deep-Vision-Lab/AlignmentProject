import torch
import random
import torch.nn.functional as F  # Added for interpolate and conv2d
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from newDataSet import TextLineModern, window_size

from Parameters import *
from DiffNWAlgo import *


_default_data_dir = f'DataSet/Synthetic_{lang}_100000'


# Plain ToTensor + Resize. Inputs are NEVER differentiated wrt; setting
# requires_grad on them just wastes GPU memory on the input grad buffer.
transform = transforms.Compose([
    transforms.Resize((128, 1024)),
    transforms.ToTensor(),
])


# ---------- Tunables for the DataLoader (override via env vars) ----------
_num_workers = int(os.environ.get('DATALOADER_NUM_WORKERS', 4))
_pin_memory = torch.cuda.is_available()
_persistent_workers = _num_workers > 0
_prefetch_factor = int(os.environ.get('DATALOADER_PREFETCH', 4)) if _num_workers > 0 else None


def _make_loader(ds, shuffle):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_persistent_workers,
        drop_last=False,
    )
    if _prefetch_factor is not None:
        kwargs['prefetch_factor'] = _prefetch_factor
    return DataLoader(ds, **kwargs)


def build_dataloaders(data_dir=None):
    """Build train/valid/test DataLoaders for the given dataset directory.

    Args:
        data_dir: Path to the dataset root (must contain images/ and texts/ subdirs).
                  Defaults to DataSet/Synthetic_{lang}_{num_samples} from Parameters.py.

    Returns:
        (train_dataloader, valid_dataloader, test_dataloader)
    """
    if data_dir is None:
        data_dir = _default_data_dir

    dataset_paths = {
        "images": os.path.join(data_dir, "images"),
        "matrices": os.path.join(data_dir, "matrices"),
        "diffNWmatrices": os.path.join(data_dir, "diffNWmatrices"),
        "similarity_matrices": os.path.join(data_dir, "similarity_matrices"),
        "texts": os.path.join(data_dir, "texts"),
    }

    full_dataset = TextLineModern(new_dataset=dataset_paths, transform=transform, num_samples_override=num_samples)

    train_sz = int(0.6 * len(full_dataset))
    valid_sz = int(0.2 * len(full_dataset))
    test_sz = len(full_dataset) - train_sz - valid_sz

    train_ds, valid_ds, test_ds = random_split(full_dataset, [train_sz, valid_sz, test_sz])

    return _make_loader(train_ds, shuffle=True), _make_loader(valid_ds, shuffle=False), _make_loader(test_ds, shuffle=False)


# Module-level defaults (used when newDataLoader is imported without a custom data_dir)
new_dataset = {
    "images": os.path.join(_default_data_dir, "images"),
    "matrices": os.path.join(_default_data_dir, "matrices"),
    "diffNWmatrices": os.path.join(_default_data_dir, "diffNWmatrices"),
    "similarity_matrices": os.path.join(_default_data_dir, "similarity_matrices"),
    "texts": os.path.join(_default_data_dir, "texts"),
}

full_dataset = TextLineModern(new_dataset=new_dataset, transform=transform)

train_size = int(0.6 * len(full_dataset))
valid_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])


# ---- Hard Negative Generators (applied to the POSITIVE text) ----

# Dot-confusion table for Arabic letters (same base glyph, different dots)
_DOT_CONFUSIONS = {
    "ب": ["ت", "ث", "ن", "ي"], "ت": ["ب", "ث", "ن", "ي"],
    "ث": ["ب", "ت", "ن", "ي"], "ن": ["ب", "ت", "ث", "ي"],
    "ي": ["ب", "ت", "ث", "ن"], "ج": ["ح", "خ"], "ح": ["ج", "خ"],
    "خ": ["ج", "ح"], "د": ["ذ"], "ذ": ["د"], "ر": ["ز"], "ز": ["ر"],
    "س": ["ش"], "ش": ["س"], "ص": ["ض"], "ض": ["ص"],
    "ط": ["ظ"], "ظ": ["ط"], "ع": ["غ"], "غ": ["ع"],
    "ف": ["ق"], "ق": ["ف"],
}
_ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")


def _hard_neg_crop(text):
    words = text.split()
    if len(words) < 2:
        return text
    cutoff = max(1, len(words) // 2)
    return ' '.join(words[:cutoff])


def _hard_neg_drop(text):
    words = text.split()
    if len(words) < 2:
        return text
    idx = random.randint(0, len(words) - 1)
    return ' '.join(words[:idx] + words[idx + 1:])


def _hard_neg_shuffle(text):
    words = text.split()
    if len(words) < 2:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return ' '.join(words)


def _hard_neg_dot_confusion(text, p=0.25):
    """Replace confusable Arabic letters at probability p; ensure ≥1 change."""
    chars = list(text)
    changed = False
    for i, ch in enumerate(chars):
        if ch in _DOT_CONFUSIONS and random.random() < p:
            chars[i] = random.choice(_DOT_CONFUSIONS[ch])
            changed = True
    if not changed:
        confusable = [(i, c) for i, c in enumerate(chars) if c in _DOT_CONFUSIONS]
        if confusable:
            idx, ch = random.choice(confusable)
            chars[idx] = random.choice(_DOT_CONFUSIONS[ch])
    return ''.join(chars)


def _hard_neg_word_shuffle(text):
    words = text.split()
    if len(words) < 2:
        return text
    random.shuffle(words)
    return ' '.join(words)


def _hard_neg_same_length_random(text):
    return ''.join(random.choice(_ARABIC_LETTERS) if ch != ' ' else ' ' for ch in text)


_hard_neg_fns  = [_hard_neg_crop, _hard_neg_drop, _hard_neg_shuffle]
_lc_neg_fns    = [_hard_neg_word_shuffle, _hard_neg_same_length_random]


def _ensure_different(neg, pos):
    """Guarantee the negative text differs from the positive (at least one char)."""
    if neg.strip() != pos.strip():
        return neg
    chars = list(pos.strip())
    if len(chars) > 1:
        import random as _rnd
        _rnd.shuffle(chars)
        candidate = ''.join(chars)
        if candidate.strip() != pos.strip():
            return candidate
    if len(pos.strip()) > 1:
        return pos.strip()[:-1]
    return pos + "‌"  # zero-width non-joiner as last resort


def _maybe_crop(text):
    """Randomly crop negative text to 50-100% of its length."""
    if random.random() < 0.3 and len(text) > 3:
        crop_ratio = random.uniform(0.5, 0.9)
        crop_len = max(2, int(len(text) * crop_ratio))
        start = random.randint(0, len(text) - crop_len)
        return text[start:start + crop_len]
    return text


def _build_negatives_for_sample(pos_text, all_pos_texts, sample_idx, mode):
    """Build num_negatives negative strings for one positive using selected mode."""
    mode = mode.lower()
    sample_negs = []

    if mode == "mixed":
        num_hard = min(2, num_negatives)
        num_random = num_negatives - num_hard
        for _ in range(num_hard):
            fn  = random.choice(_hard_neg_fns)
            neg = _ensure_different(fn(pos_text), pos_text)
            sample_negs.append(neg)
        available = [j for j, t in enumerate(all_pos_texts) if j != sample_idx]
        if not available:
            available = [sample_idx]
        sampled = (random.sample(available, num_random)
                   if len(available) >= num_random
                   else [random.choice(available) for _ in range(num_random)])
        for j in sampled:
            neg = _ensure_different(_maybe_crop(all_pos_texts[j]), pos_text)
            sample_negs.append(neg)

    elif mode == "length_controlled":
        fns   = _lc_neg_fns + [_hard_neg_word_shuffle]
        names = ["word_shuffled", "same_length_random", "word_shuffled2"]
        num_lc = min(len(fns), num_negatives)
        for fn in fns[:num_lc]:
            neg = _ensure_different(fn(pos_text), pos_text)
            sample_negs.append(neg)
        remaining = num_negatives - len(sample_negs)
        for _ in range(remaining):
            neg = _ensure_different(_hard_neg_word_shuffle(pos_text), pos_text)
            sample_negs.append(neg)

    elif mode == "dot_confusion":
        for _ in range(num_negatives):
            neg = _ensure_different(_hard_neg_dot_confusion(pos_text), pos_text)
            sample_negs.append(neg)

    elif mode == "same_length_random":
        for _ in range(num_negatives):
            neg = _ensure_different(_hard_neg_same_length_random(pos_text), pos_text)
            sample_negs.append(neg)

    elif mode == "shuffle_only":
        for _ in range(num_negatives):
            neg = _ensure_different(_hard_neg_word_shuffle(pos_text), pos_text)
            sample_negs.append(neg)

    else:
        # Unknown mode — fall back to mixed
        return _build_negatives_for_sample(pos_text, all_pos_texts, sample_idx, "mixed")

    return sample_negs


def custom_collate_fn(batch):
    """
    Collate function for contrastive learning with in-batch negatives.

    Runs entirely on CPU inside dataloader workers; the training loop is
    responsible for the H2D copy (with non_blocking / pinned memory).
    Negative mode is read from Parameters.negative_mode at collation time.
    """
    texts1, images1 = zip(*batch)

    images    = torch.stack(images1, dim=0)
    pos_texts = list(texts1)

    try:
        from Parameters import negative_mode as _neg_mode
    except ImportError:
        _neg_mode = "mixed"

    neg_texts = [
        _build_negatives_for_sample(pos_texts[i], pos_texts, i, _neg_mode)
        for i in range(len(pos_texts))
    ]

    return images, pos_texts, neg_texts


# Optional helper kept for callers that need padded matrices
def pad_matrices(matrices, smooth=False, kernel_size=5, sigma=1.0):
    if not matrices:
        return torch.empty(0)

    device = matrices[0].device
    max_dim = max(max(mat.shape) for mat in matrices)

    gaussian_kernel_2d = None
    if smooth:
        _x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=torch.float32)
        _gauss1d = torch.exp(-_x.pow(2) / (2 * sigma**2))
        _gauss1d /= _gauss1d.sum()
        gaussian_kernel_2d = torch.outer(_gauss1d, _gauss1d).unsqueeze(0).unsqueeze(0)

    processed_matrices = []
    for mat in matrices:
        mat = mat.to(device)
        mat_unsqueezed = mat.unsqueeze(0).unsqueeze(0)
        processed_mat = F.interpolate(mat_unsqueezed, size=(max_dim, max_dim), mode='nearest')
        if smooth and gaussian_kernel_2d is not None:
            current_kernel = gaussian_kernel_2d.to(processed_mat.device)
            padding = kernel_size // 2
            processed_mat = F.conv2d(processed_mat, current_kernel, padding=padding)
        processed_matrices.append(processed_mat.squeeze(0).squeeze(0))

    return torch.stack(processed_matrices, dim=0).to(device)


# Create DataLoaders for training and testing
train_dataloader = _make_loader(train_dataset, shuffle=True)
valid_dataloader = _make_loader(valid_dataset, shuffle=False)
test_dataloader = _make_loader(test_dataset, shuffle=False)


if __name__ == "__main__":
    for batch in train_dataloader:
        images, pos_texts, neg_texts = batch
        print(f'Batch images shape: {images.shape}')
        print(f'Pos texts: {len(pos_texts)}')
        print(f'Neg texts per sample: {len(neg_texts[0])} (num_negatives={num_negatives})')
        break
