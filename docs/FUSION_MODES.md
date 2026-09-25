# Local/context fusion on current-arch-model

These choices change only fusion, not the ResNet, Transformer, physical window
order, preprocessing, character codebook, losses or optimizer. The default is
`--fusion-mode concat --use-gated-fusion 0`. No new attention is used.

For the compact `resnet18_128d_5l_1h_no_pos` experiment, L and C are both
`[B,T,128]` (T=63 for a 1024-pixel line, width 32 and stride 16). No dimension
alignment projections are needed. Each output uses only the corresponding
local/context pair at index i:

| CLI | Raw fused vector Z_i |
| --- | --- |
| concat, gate 0 | Linear(256,128)([L_i; C_i]), the unchanged original module |
| sum, gate 0 | L_i + C_i |
| sum, gate 1 | L_i + alpha_i * C_i |

The feature-wise gate is `alpha_i = sigmoid(Linear(128,128)(GELU(Linear(256,128)([L_i;C_i]))))`.
It has **49,408 trainable parameters**, including biases, uses fresh ordinary
PyTorch Linear initialization, and has no attention or dropout. The local
residual is unconditional; this is **not** alpha*L+(1-alpha)*C. The compact
model's existing LayerNorm(128) then produces pre-L2 h; existing L2 normalization
produces z for cosine/DTW. SIGReg still uses h, not z. Concat+gate 1 is rejected.

The previous 192-D backend keeps its original concat MLP and normalization;
the sum heads retain its existing output-normalization convention too.
Checkpoints record `fusion_mode`, `use_gated_fusion`, the gate MLP/count, and
actual model parameter counts. Evaluation rebuilds from checkpoint settings
and loads strictly. Older concat checkpoint metadata (including missing fields)
remains compatible. Changing mode/gate metadata on saved weights fails strictly;
the compact experiment still disallows full-model/optimizer initialization from
previous runs. Start separate runs for each mode.

## Commands (from the repository root)

CPU forward/backward checks, using the actual compact ResNet/Transformer and
existing DTW/SIGReg on random fixture images, not a training run:

```bash
OMP_NUM_THREADS=2 python -m pytest -q -s tests/test_fusion_modes.py
```

Single-device training, using the unchanged compact launcher's dataset, loss,
and optimization settings (choose a fresh JOB_NAME every time):

```bash
JOB_NAME=fusion_concat_gate0_run1 bash scripts/train_resnet18_128d_5l_1h_no_pos.sh --fusion-mode concat --use-gated-fusion 0
JOB_NAME=fusion_sum_gate0_run1 bash scripts/train_resnet18_128d_5l_1h_no_pos.sh --fusion-mode sum --use-gated-fusion 0
JOB_NAME=fusion_sum_gate1_run1 bash scripts/train_resnet18_128d_5l_1h_no_pos.sh --fusion-mode sum --use-gated-fusion 1 --gate-diagnostics 1
```

Two-4090 SLURM submissions (new launcher delegates all existing settings to the
compact launcher; job-ID-qualified output names avoid historical runs):

```bash
mkdir -p out
FUSION_MODE=concat USE_GATED_FUSION=0 sbatch scripts/train.sbatch
FUSION_MODE=sum USE_GATED_FUSION=0 sbatch scripts/train.sbatch
FUSION_MODE=sum USE_GATED_FUSION=1 sbatch scripts/train.sbatch --gate-diagnostics 1
```

`DATASET=/absolute/path/to/ArabicDataset` and `PYTHON_BIN=/path/to/python` may
override the existing launcher defaults. `scripts/train.sbatch` did not exist
on this branch before this change; it copies resource requests from the
existing compact-model sbatch launcher, without migrating the simplified branch.

The model startup log prints fusion mode and enabled/disabled gate status.
Optional `--gate-diagnostics 1` logs `GATE` records on rank zero: mean, population
std, min, max, and percentages in [0,.25), [.25,.5), [.5,.75), [.75,1]. These
are detached, valid-window-only **rank-local** statistics, not global DDP
estimates. They add no loss and are disabled by default. Gate values are feature
weights, not alignment probabilities. No quality claim follows from a smoke
test: compare separately trained modes using held-out evaluation.
