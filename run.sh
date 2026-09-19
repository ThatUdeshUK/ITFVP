#!/usr/bin/env bash
# Install uv if needed, sync dependencies if needed, then launch the GUI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
    echo "Setting up the virtual environment..."
    uv sync
fi

exec uv run python main.py
