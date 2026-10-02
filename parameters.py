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
    cnn_layers: int = 3  # simple-CNN depth only; ResNet18 stays fixed
    transformer_type: str = 'tiny'
    transformer_layers: int = 0  # 0 uses the preset depth.
    transformer_heads: int = 0  # 0 uses the preset head count.
    use_positional_encoding: bool = False
    embedding_dim: int = 128
    local_dropout: float = .10
    transformer_dropout: float = 0.
    fusion_mode: str = 'concat'
    use_gated_fusion: int = 0
    rtl: bool = True
    text_embedding_seed: int = 1234
    text_vocab_size: int = 4096
    positive_dtw_weight: float = 1.0
    negative_dtw_weight: float = 0.0
    negative_margin: float = .20
    # Weight is the single enable switch: zero means no generation/alignment.
    negative_count: int = 3
    hard_negative_k: int = 2  # lowest alignment energies among generated candidates
    negative_loss_type: str = 'ranking'  # ranking/absolute=legacy; hybrid_typed=type-specific supervision
    strong_negative_count: int = 2
    local_substitution_count: int = 1
    order_negative_count: int = 1
    strong_negative_weight: float = .5
    wrong_letter_weight: float = .5
    order_negative_weight: float = .25
    strong_negative_severity: float = .75
    local_substitution_severity: float = .30
    order_margin: float = .20
    # Initial diagnostic values: 16 validation lines from the available
    # 2026-10-01 checkpoint had median aligned positive NLL 1.08 and 64
    # corrupted negatives had median NLL 1.69 (uniform 37-letter NLL = 3.61).
    # Tune on validation only; these are not claimed optimal/calibrated.
    negative_target_min: float = 1.5
    negative_target_max: float = 2.5
    negative_target_mode: str = 'fixed'  # old checkpoints retain fixed thresholds
    negative_target_min_ratio: float = .70
    negative_target_max_ratio: float = 1.00
    negative_softness: float = .10  # NLL-scale softplus hinge smoothing
    ranking_aux_weight: float = 0.0  # only with absolute; not multiplied by negative_dtw_weight
    wrong_letter_unlikelihood_weight: float = 0.0  # substitution-only positive-occupancy auxiliary
    negative_operations: str = 'substitute,adjacent,blocks,words,shift,shuffle'
    negative_severity: float = .35
    negative_seed: int = 42
    negative_warmup_epochs: int = 0
    negative_curriculum_epochs: int = 5
    alignment_objective: str = 'dtw'  # optional standalone CTC experiment
    ctc_blank_logit: float = 0.
    alphabet_inventory: str = ''  # empty preserves historical per-line alphabet extension
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
    dtw_normalization: str = 'legacy'  # old total/(T+L); explicit aligned_mean for new runs
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.
    num_workers: int = 4
    use_amp: bool = True
    seed: int = 42


def validate_objective(config):
    import math
    if config.alignment_objective not in {'dtw', 'ctc'}:
        raise ValueError('alignment_objective must be dtw or ctc')
    if config.dtw_cost_mode not in {'full_alphabet_nll', 'cosine'}:
        raise ValueError('Unknown dtw_cost_mode')
    if config.dtw_normalization not in {'legacy', 'aligned_mean'}:
        raise ValueError('dtw_normalization must be legacy or aligned_mean')
    if config.negative_loss_type not in {'ranking', 'absolute', 'hybrid_typed'}:
        raise ValueError('negative_loss_type must be ranking, absolute or hybrid_typed')
    if not 1 <= config.hard_negative_k <= config.negative_count:
        raise ValueError('Require 1 <= hard_negative_k <= negative_count')
    if config.negative_target_mode not in {'fixed', 'uniform_ratio'}:
        raise ValueError('negative_target_mode must be fixed or uniform_ratio')
    if config.negative_loss_type == 'hybrid_typed':
        if config.negative_dtw_weight or config.ranking_aux_weight or config.wrong_letter_unlikelihood_weight:
            raise ValueError('hybrid_typed uses its own component weights; set legacy negative weights to zero')
        if sum((config.strong_negative_count, config.local_substitution_count,
                config.order_negative_count)) != config.negative_count:
            raise ValueError('Typed negative counts must sum to negative_count')
        if any(v < 0 for v in (config.strong_negative_count, config.local_substitution_count,
                               config.order_negative_count)):
            raise ValueError('Typed negative counts must be nonnegative')
        if (not all(math.isfinite(v) and v >= 0 for v in
                    (config.strong_negative_weight, config.wrong_letter_weight,
                     config.order_negative_weight, config.order_margin))):
            raise ValueError('Typed weights and order margin must be finite and nonnegative')
        if not .70 <= config.strong_negative_severity <= .90:
            raise ValueError('strong_negative_severity must be in [0.70, 0.90]')
        if not .20 <= config.local_substitution_severity <= .35:
            raise ValueError('local_substitution_severity must be in [0.20, 0.35]')
        if (config.alignment_objective != 'dtw' or config.dtw_cost_mode != 'full_alphabet_nll'
                or config.dtw_normalization != 'aligned_mean' or config.negative_target_mode != 'uniform_ratio'):
            raise ValueError('hybrid_typed requires aligned_mean full_alphabet_nll DTW and uniform_ratio targets')
        if config.negative_curriculum_epochs:
            raise ValueError('hybrid_typed requires negative_curriculum_epochs=0')
    if not math.isfinite(config.negative_softness) or config.negative_softness <= 0:
        raise ValueError('negative_softness must be finite and positive')
    if config.negative_target_mode == 'fixed':
        if (not all(math.isfinite(x) for x in (config.negative_target_min, config.negative_target_max))
                or not 0 < config.negative_target_min < config.negative_target_max):
            raise ValueError('Fixed targets require 0 < min < max')
    elif (not all(math.isfinite(x) for x in
                  (config.negative_target_min_ratio, config.negative_target_max_ratio))
          or not 0 < config.negative_target_min_ratio < config.negative_target_max_ratio <= 1.5):
        raise ValueError('Uniform-ratio targets require 0 < min_ratio < max_ratio <= 1.5')
    if any(not math.isfinite(x) or x < 0 for x in
           (config.ranking_aux_weight, config.wrong_letter_unlikelihood_weight)):
        raise ValueError('Auxiliary weights must be finite and nonnegative')
    if config.negative_dtw_weight == 0 and (config.ranking_aux_weight or config.wrong_letter_unlikelihood_weight):
        raise ValueError('Negative auxiliaries require negative_dtw_weight > 0')
    if config.negative_loss_type == 'absolute' and (config.alignment_objective != 'dtw'
                                                   or config.dtw_cost_mode != 'full_alphabet_nll'
                                                   or config.dtw_normalization != 'aligned_mean'):
        raise ValueError('Absolute negative loss requires DTW, full_alphabet_nll and aligned_mean normalization')
    if config.negative_loss_type == 'ranking' and (config.ranking_aux_weight or config.wrong_letter_unlikelihood_weight):
        raise ValueError('Auxiliary ranking/wrong-letter weights require absolute negative loss')
    if config.wrong_letter_unlikelihood_weight and (config.negative_loss_type != 'absolute'
                                                   or config.alignment_objective != 'dtw'):
        raise ValueError('Wrong-letter unlikelihood requires absolute DTW negative loss')
    if any(not math.isfinite(v) or v < 0 for v in (config.negative_dtw_weight, config.negative_margin, config.position_prior)):
        raise ValueError('Negative weight, margin and position prior must be finite and nonnegative')
    if not math.isfinite(config.competition_temperature) or config.competition_temperature <= 0:
        raise ValueError('competition_temperature must be positive')
    if not math.isfinite(config.ctc_blank_logit):
        raise ValueError('ctc_blank_logit must be finite')
    if config.alphabet_inventory:
        from text_embedding import clean_letters
        if ''.join(clean_letters(config.alphabet_inventory)) != config.alphabet_inventory or len(set(config.alphabet_inventory)) != len(config.alphabet_inventory):
            raise ValueError('alphabet_inventory must be unique normalized Arabic letters')
    if config.negative_count < 1 or not 0 < config.negative_severity <= 1:
        raise ValueError('negative_count >= 1 and negative_severity in (0,1] required')
    if min(config.negative_warmup_epochs, config.negative_curriculum_epochs) < 0:
        raise ValueError('Negative warmup/curriculum epochs must be nonnegative')
    operations = set(config.negative_operations.split(','))
    if not operations or not operations <= {'substitute', 'adjacent', 'blocks', 'words', 'shift', 'shuffle'}:
        raise ValueError('Unknown/empty negative_operations')
    if config.alignment_objective == 'ctc' and (config.position_prior != 0 or config.dtw_cost_mode != 'full_alphabet_nll'):
        raise ValueError('CTC requires full_alphabet_nll and --position-prior 0 (no DTW prior)')


def negatives_enabled(config):
    if config.negative_loss_type == 'hybrid_typed':
        return bool(config.strong_negative_weight or config.wrong_letter_weight or config.order_negative_weight)
    return bool(config.negative_dtw_weight)
