"""Configuration for the standalone pipeline. No architecture environment variables."""
from dataclasses import dataclass


@dataclass
class Config:
    dataset_type: str = 'auto'
    image_height: int = 128
    image_width: int = 1024
    grayscale: bool = True
    crop: str = 'auto'
    binarize: bool = False
    paired: bool = False
    train_ratio: float = .8
    val_ratio: float = .1
    test_ratio: float = .1
    split_mode: str = 'group'
    split_seed: int = 42
    augmentation: bool = True
    window_width: int = 32
    window_stride: int = 16
    cnn_type: str = 'resnet18'
    cnn_pretrained: bool = True
    transformer_type: str = 'tiny'
    use_positional_encoding: bool = False
    embedding_dim: int = 128
    local_dropout: float = .10
    transformer_dropout: float = 0.
    rtl: bool = True
    text_embedding_seed: int = 1234
    text_vocab_size: int = 4096
    positive_dtw_weight: float = 1.0
    negative_dtw_weight: float = 0.0
    negative_margin: float = .20
    sigreg_weight: float = 0.0
    sigreg_sketch_dim: int = 1024
    sigreg_num_knots: int = 17
    sigreg_min_samples: int = 32
    dtw_gamma: float = .05
    vertical_penalty: float = .05
    horizontal_penalty: float = .30
    position_prior: float = .15
    competition_temperature: float = .10
    disable_horizontal_when_feasible: bool = True
    dtw_cost_mode: str = 'full_alphabet_nll'
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.
    num_workers: int = 4
    use_amp: bool = True
    seed: int = 42
