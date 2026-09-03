#!/usr/bin/env bash
#
# Iridis environment setup.  Loads the conda module and activates the
# project environment, creating it on first use.
#
# MUST be sourced, not executed, because `conda activate` changes the
# current shell:
#
#     source ./_setup.sh
#
# Override the environment name with:  ENV_NAME=other source ./_setup.sh

ENV_NAME="${ENV_NAME:-grn}"
PY_VERSION="${PY_VERSION:-3.12}"

# Resolve the project directory whether sourced or executed.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    PROJECT_DIR="$(pwd)"
fi

module load conda/python3 || {
    echo "ERROR: could not 'module load conda/python3'." >&2
    echo "       Run 'module avail conda' and use the name listed there." >&2
    return 1 2>/dev/null || exit 1
}

# Create the environment the first time only.
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating conda environment '${ENV_NAME}' (first run only) ..."

    if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
        conda create -n "${ENV_NAME}" "python=${PY_VERSION}" \
              --file "${PROJECT_DIR}/requirements.txt" -y \
            || conda create -n "${ENV_NAME}" "python=${PY_VERSION}" numpy matplotlib -y
    else
        conda create -n "${ENV_NAME}" "python=${PY_VERSION}" numpy matplotlib -y
    fi
fi

conda activate "${ENV_NAME}" || {
    echo "ERROR: could not activate '${ENV_NAME}'." >&2
    return 1 2>/dev/null || exit 1
}

python -c "import numpy, matplotlib; print('OK: numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)" || {
    echo "ERROR: numpy/matplotlib not importable inside '${ENV_NAME}'." >&2
    echo "       Try: conda activate ${ENV_NAME} && pip install numpy matplotlib" >&2
    return 1 2>/dev/null || exit 1
}
