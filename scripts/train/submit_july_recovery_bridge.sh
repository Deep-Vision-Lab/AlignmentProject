#!/usr/bin/env bash
# Safe submission wrapper for run_july_recovery_bridge.sh.
#
# Slurm copies submitted scripts into its spool directory before execution, so
# BASH_SOURCE[0] on the compute node is not the repository script path.  Export
# the real checkout root explicitly before the inner launcher calls sbatch.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
export PROJECT_DIR="${ROOT}"

exec bash "${ROOT}/scripts/train/run_july_recovery_bridge.sh"
