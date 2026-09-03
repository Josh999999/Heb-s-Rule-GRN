#!/bin/bash -l
#SBATCH --job-name=grn-exp4
#SBATCH -A ecsstudents
#SBATCH -p ecsstudents_l4
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jm2e25@soton.ac.uk
#
# Run Experiment 4.
#     sbatch job_experiment4.sh
#
# Pure NumPy: no --gres=gpu requested.
# ecsstudents_l4 has a 24 h wall-clock limit, so --time cannot exceed 24:00:00.

module load conda/python3

# Define the `conda activate` shell function (module load alone does not).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-grn}"

# Work from the directory holding the Experiment*.py files.
# Set PROJECT_DIR if the .py files live somewhere other than this script's folder,
# e.g. PROJECT_DIR=.. when these scripts sit in a scripts/ subfolder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR:-$SCRIPT_DIR}" || exit 1

if [ ! -f "Experiment4.py" ]; then
    echo "ERROR: Experiment4.py not found in $(pwd)" >&2
    echo "       Set PROJECT_DIR to the folder containing it, e.g." >&2
    echo "       sbatch --export=ALL,PROJECT_DIR=$HOME/Heb-s-Rule-GRN job_experiment4.sh" >&2
    exit 1
fi

mkdir -p out

echo "host      : $(hostname)"
echo "workdir   : $(pwd)"
echo "started   : $(date)"
echo "python    : $(which python)"
python -c "import numpy, matplotlib; print('numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"
echo "----------------------------------------------------------------------"

python Experiment4.py
status=$?

echo "----------------------------------------------------------------------"
echo "finished  : $(date)  (exit ${status})"
exit ${status}
