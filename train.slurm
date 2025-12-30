#!/bin/bash
#SBATCH --job-name=soccernet-train
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gres=gpu:rtx30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=evaluation/training_logs/train_%j.out
#SBATCH --error=evaluation/training_logs/train_%j.err

set -o errexit
set -o nounset

module purge
module load Miniconda3/22.11.1-1
module load CUDA/12.8.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate SoccerNet311

echo "Python path: $(which python)"
python -c "import torch; print('CUDA available?', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

cd ~/thumbnail-selection
python src/training/train.py "$@"

