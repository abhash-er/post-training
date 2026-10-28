#!/bin/bash
#SBATCH --job-name=dolci-en-verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=/gpfs/scratch/ehpc532/eit234648/post-training/data_prep/logs/verify_%j.out
#
# Language check of Dolci-Instruct-SFT (see verify_dolci_lowconf.py). CPU only;
# runs here because the login-node process killer stops it after ~5 min.
# Uses the system python3 + ~/.local (fasttext-wheel, pyarrow, numpy<2).

set -uo pipefail
cd /gpfs/scratch/ehpc532/eit234648
/usr/bin/python3 --version
/usr/bin/python3 -u post-training/data_prep/verify_dolci_lowconf.py
