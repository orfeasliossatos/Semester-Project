#!/bin/sh

#SBATCH --mail-user=oiliossatos@gmail.com
#SBATCH --mail-type=end
#SBATCH --time=06:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition slurm-cluster
#SBATCH --job-name=week2
#SBATCH --output=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/test.%J.out
#SBATCH --error=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/test.%J.err

# Navigate to job location in directory
cd ~/Documents/Semester-Project/Tasks/

# Print the start date
echo "execution started on $(date)"

# Activate test environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test

# Run python script
srun python week2.py -min 4 -max 14 -f week2_out.json -e 100 -p 2

# Print end date
echo "execution finished at $(date)"