#!/bin/bash
set -euo pipefail

exec srun uv run --locked "$@"
