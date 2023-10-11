#!/bin/sh

# My email address
#SBATCH --mail-user=oiliossatos@gmail.com
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --nodelist=iscpc88.epfl.ch
#SBATCH --cpus-per-task=4
#SBATCH --partition slurm-gpu
#SBATCH --job-name=install

# Where do the output logs go?
#SBATCH --output=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/%J.out
#SBATCH --error=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/%J.err
#SBATCH --gres=gpu:1

# Navigate to job location in directory
cd ~/Documents/Semester-Project/Tasks/

# Print the name of the machine is running on
echo "executed on $HOSTNAME"

# Print the start date
echo "execution started on $(date)"

# Activate test environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate olenv

# Get information
python scripts/cuda_available.py

# Print end date
echo "execution finished at $(date)"
