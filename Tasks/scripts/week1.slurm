#!/bin/sh

#SBATCH --mail-user=oiliossatos@gmail.com
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition slurm-gpu
#SBATCH --job-name=week1
#SBATCH --output=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/test.%J.out
#SBATCH --error=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/test.%J.err
#SBATCH --gres=gpu:1

# Navigate to job location in directory
cd ~/Documents/Semester-Project/Tasks/

# Print the start date
echo "execution started on $(date)"

# Activate test environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test

# Experiment variables
min_mag=2
max_mag=6
splits=5
max_epochs=10
norm=2
width=6
delete_results=False
filepath="results/week1/p${norm}_l${width}_e${max_epochs}.pkl"

# Run python script
python tr_size_vs_te_acc.py -actv "Quad" -arch "DumbCNN" -min $min_mag -max $max_mag -f $filepath -s $splits -e $max_epochs -p $norm -l $width -del $delete_results

# Print end date
echo "execution finished at $(date)"

