#!/usr/bin/env bash
#
# Run Experiment 3. Installs requirements into .venv on first use,
# then executes Experiment3.py, writing figures to Experiment3/.
#
# Usage:  ./run_experiment3.sh

source "$(dirname "${BASH_SOURCE[0]}")/_setup.sh"

run_experiment 3
