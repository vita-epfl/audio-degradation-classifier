#!/bin/bash
#SBATCH --job-name=audio-degradation-classifier
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --partition=l40s
#SBATCH --output=/work/vita/alefevre/programs/audio-degradation-classifier/slurm/logs/%j.out
#SBATCH --error=/work/vita/alefevre/programs/audio-degradation-classifier/slurm/logs/%j.err


mkdir -p /work/vita/alefevre/programs/audio-degradation-classifier/slurm/logs

# Create a directory for local installations
mkdir -p ~/local/sox > /dev/null 2>&1
cd ~/local > /dev/null 2>&1

# Download and extract SoX source (silently)
wget -q https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz
tar -xzf sox-14.4.2.tar.gz > /dev/null 2>&1
cd sox-14.4.2 > /dev/null 2>&1

# Configure and install to your local directory (silently)
./configure --prefix=$HOME/local/sox > /dev/null 2>&1
make > /dev/null 2>&1
make install > /dev/null 2>&1

# Add to your PATH and LD_LIBRARY_PATH in your SLURM script
export PATH=$HOME/local/sox/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/sox/lib:$LD_LIBRARY_PATH

export WANDB_API_KEY=6bc7bb9456313ef80ea74ee7877c4c927de72220
export PROJECT_NAME="audio-degradation-classifier"

cd /work/vita/alefevre/programs/audio-degradation-classifier

source /work/vita/alefevre/programs/audio-degradation-classifier/venv/bin/activate

echo STARTING AT `date`

srun --gpu-bind=closest /bin/bash -c "python train.py --config config_scitas.yaml --effects-config effects_config.yaml"

echo FINISHED at `date`