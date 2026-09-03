#!/usr/bin/env bash
#
# Run Experiment 4. Installs requirements into .venv on first use,
# then executes Experiment4.py, writing figures to Experiment4/.
#
# Usage:  ./run_experiment4.sh

source "$(dirname "${BASH_SOURCE[0]}")/_setup.sh"

run_experiment 4
