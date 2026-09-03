#!/usr/bin/env bash
#
# Shared setup, sourced by each run_experimentN.sh script.
# Creates a local virtual environment and pip-installs requirements.txt
# the first time (and again whenever requirements.txt changes).
#
# Override the interpreter with:  PYTHON=python3.11 ./run_experiment1.sh
# Skip the venv entirely with:    USE_VENV=0 ./run_experiment1.sh

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python3}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="$PROJECT_DIR/.venv"
STAMP="$VENV_DIR/.requirements.sha"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found. Install Python 3 or set PYTHON=..." >&2
    exit 1
fi

if [ "$USE_VENV" = "1" ]; then
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in .venv ..."
        if ! "$PYTHON" -m venv "$VENV_DIR"; then
            echo "ERROR: could not create a venv." >&2
            echo "       On Debian/Ubuntu try: sudo apt install python3-venv" >&2
            echo "       Or re-run with:       USE_VENV=0 $0" >&2
            exit 1
        fi
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PY="python"
else
    PY="$PYTHON"
fi

# Install requirements only when they have changed since the last install.
if [ -f requirements.txt ]; then
    if command -v sha256sum >/dev/null 2>&1; then
        want="$(sha256sum requirements.txt | cut -d' ' -f1)"
    else
        want="$(shasum -a 256 requirements.txt | cut -d' ' -f1)"
    fi
    have="$(cat "$STAMP" 2>/dev/null || echo none)"

    if [ "$want" != "$have" ]; then
        echo "Installing requirements ..."
        "$PY" -m pip install --quiet --upgrade pip
        if "$PY" -m pip install --quiet -r requirements.txt; then
            [ "$USE_VENV" = "1" ] && echo "$want" > "$STAMP"
            echo "Requirements installed."
        else
            echo "ERROR: pip install failed." >&2
            exit 1
        fi
    fi
fi

# Fail early with a clear message if the imports are still missing.
if ! "$PY" -c "import numpy, matplotlib" >/dev/null 2>&1; then
    echo "ERROR: numpy/matplotlib are not importable after install." >&2
    exit 1
fi

# run_experiment() <n> -- run one experiment and report its figures.
run_experiment() {
    local n="$1"
    local script="Experiment${n}.py"
    local folder="Experiment${n}"

    echo
    echo "======================================================================"
    echo " Experiment ${n}"
    echo "======================================================================"

    if [ ! -f "$script" ]; then
        echo "ERROR: $script not found in $PROJECT_DIR" >&2
        return 1
    fi

    local start=$SECONDS
    local status=0
    "$PY" "$script" || status=$?

    # Experiment 3 writes some image strips to the working directory.
    mkdir -p "$folder"
    for stray in Figure*_imgstrip.png; do
        [ -e "$stray" ] && mv "$stray" "$folder/"
    done

    local count
    count=$(ls "$folder"/*.png 2>/dev/null | wc -l | tr -d ' ')

    if [ "$status" -eq 0 ]; then
        echo "  finished in $(( SECONDS - start ))s - ${count} figures in ${folder}/"
    else
        echo "  FAILED (exit ${status}) after $(( SECONDS - start ))s - ${count} figures in ${folder}/" >&2
    fi

    return "$status"
}
