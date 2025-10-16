#!/bin/bash
#SBATCH --job-name=soccernet-train
#SBATCH --account=ec12
#SBATCH --partition=accel            # or accel_long if you have longer jobs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -o errexit
set -o nounset

module purge
module load Miniconda3/22.11.1-1
module load CUDA/12.8.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate SoccerNet311

echo "Python path: $(which python)"
echo "CUDA available?"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd ~/thumbnail-selection
python train.py "$@"

