#!/usr/bin/env bash
#
# Run Experiment 2. Installs requirements into .venv on first use,
# then executes Experiment2.py, writing figures to Experiment2/.
#
# Usage:  ./run_experiment2.sh

source "$(dirname "${BASH_SOURCE[0]}")/_setup.sh"

run_experiment 2
