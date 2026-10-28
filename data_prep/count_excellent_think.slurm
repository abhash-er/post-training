#!/bin/bash
#SBATCH --job-name=excellent-think
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=/gpfs/scratch/ehpc532/eit234648/post-training/data_prep/logs/excellent_think_%j.out
#
# Strict-English + <think> count of OpenEuroLLM-Excellent-SFT (see
# count_excellent_think.py). CPU only.
# Uses the system python3 + ~/.local (fasttext-wheel, pyarrow, numpy<2).

set -uo pipefail
cd /gpfs/scratch/ehpc532/eit234648
/usr/bin/python3 --version
/usr/bin/python3 -u post-training/data_prep/count_excellent_think.py
