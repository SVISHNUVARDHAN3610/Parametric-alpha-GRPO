#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
artifacts=${ALPHA_GRPO_ARTIFACTS:-/kaggle/working/alpha-grpo}
env_dir=${ALPHA_GRPO_ENV:-/kaggle/temp/alpha-grpo-venv}
export HF_HOME=${HF_HOME:-/kaggle/temp/alpha-grpo-hf}
export ALPHA_GRPO_GIT_REVISION=${ALPHA_GRPO_GIT_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}
mkdir -p "$artifacts"

if [[ ${1:-} == setup ]]; then
    uv venv --python 3.11 "$env_dir"
    uv pip install --python "$env_dir/bin/python" --torch-backend=cu129 -e '.[kaggle,evaluation,test]'
    uv pip freeze --python "$env_dir/bin/python" > "$artifacts/requirements-resolved.txt"
    "$env_dir/bin/python" -m nltk.downloader punkt punkt_tab
    exit 0
fi

exec "$env_dir/bin/alpha-grpo" "$@"
