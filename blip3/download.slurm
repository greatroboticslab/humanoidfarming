#!/bin/bash
#SBATCH --job-name=git-lfs-download
#SBATCH --output=git-lfs-download.out
#SBATCH --error=git-lfs-download.err
#SBATCH --time=2:00:00              # Request 2 hours (adjust as needed)
#SBATCH --mem=4G                    # Request memory (adjust as needed)
#SBATCH --cpus-per-task=1

# Load git-lfs if needed (depends on PSC system)
# module load git-lfs

# Change to your working directory
cd ${PROJECT}/humanoidfarming/blip3/training_data/  # or wherever you want to clone the repo

# Perform the clone
git lfs clone https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption
