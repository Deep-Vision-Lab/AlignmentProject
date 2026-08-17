#!/usr/bin/env bash
# This branch uses the same tracked model pipeline as the canonical CNN branch.
# RealSyntheticBridge V2 MUST already exist and pass validation before this script runs.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/slurm/submit_full_research_pipeline_impl.sh" "$@"
