#!/usr/bin/env bash
# Canonical Bridge V3 dataset submitter is synchronized from the CNN branch.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_bridge_v2_dataset.sh" "$@"
