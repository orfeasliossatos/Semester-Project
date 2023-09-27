#!/bin/sh

#SBATCH --mail-user=oiliossatos@gmail.com
#SBATCH --mail-type=end
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition slurm-cluster
#SBATCH --job-name=week1
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
srun python week1.py -min 2 -max 5 -f test_acc.json -s 10 -e 20

# Print end date
echo "execution finished at $(date)"

