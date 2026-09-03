#!/usr/bin/env bash
#
# Run Experiment 1. Installs requirements into .venv on first use,
# then executes Experiment1.py, writing figures to Experiment1/.
#
# Usage:  ./run_experiment1.sh

source "$(dirname "${BASH_SOURCE[0]}")/_setup.sh"

run_experiment 1
