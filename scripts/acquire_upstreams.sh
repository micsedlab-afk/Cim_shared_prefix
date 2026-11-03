#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$ROOT/_upstream"
mkdir -p "$UP"
# Requires Git credentials for private repositories. Never commit these checkouts.
clone_if_absent() {
    local name="$1"
    if [ ! -d "$UP/$name/.git" ]; then
        git clone "https://github.com/micsedlab-afk/$name.git" "$UP/$name"
    fi
}
clone_if_absent shared_prefix_cim
clone_if_absent SRAM2LLM
clone_if_absent MAGMA-gpt
clone_if_absent HBM4PiM
(
    cd "$UP/shared_prefix_cim"
    git fetch origin task/sram2llm-sinangil7nm-20260919
    git checkout --detach a8a3f7238c471da2f3b7e6fe5647f8f25b8d84c4
)
for name in shared_prefix_cim SRAM2LLM MAGMA-gpt HBM4PiM; do
    printf '%s %s\n' "$name" "$(git -C "$UP/$name" rev-parse HEAD)"
done
