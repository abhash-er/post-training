#!/bin/bash
#SBATCH --job-name=build-mixes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=/gpfs/scratch/ehpc532/eit234648/post-training/data_prep/logs/build_mixes_%j.out
#
# Build the 18 Prelude ratio-study mixes (see build_mixes.py). CPU only, runs
# inside the training container for transformers. Pools are cached, so a rerun
# after a timeout resumes at the first missing pool.

set -uo pipefail
module load singularity/4.1.5
REPO=/gpfs/scratch/ehpc532/eit234648/post-training
singularity exec --bind /gpfs/scratch/ehpc532 "$REPO/containers/torch-base" bash -c "
  export PATH=$REPO/containers/trl-venv/bin:\$PATH TOKENIZERS_PARALLELISM=false
  cd $REPO/data_prep && python -u build_mixes.py --workers 78
"
