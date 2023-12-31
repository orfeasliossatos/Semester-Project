#!/bin/sh

#SBATCH --mail-user=oiliossatos@gmail.com
#SBATCH --mail-type=end
#SBATCH --time=06:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition slurm-gpu
#SBATCH --job-name=paramfair
#SBATCH --output=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/%J.out
#SBATCH --error=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/%J.err
#SBATCH --gres=gpu:tesla:1

# Navigate to job location in directory
cd ~/Documents/Semester-Project/Tasks/

# Print the start date
echo "execution started on $(date)"

# Activate test environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tesla

# Experiment variables
min_w=3
max_w=8
max_epochs=200
norm=2
accuracy=90
delete_results=True
filepath="results/week3/p${norm}_acc${accuracy}_e${max_epochs}.pkl"

# Run python script
props=(
    0.1
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
    1
)

for i in "${props[@]}"; do
    srun python dim_vs_tr_size.py -actv "Quad" -arch "ParamFairCNN" -prop $i -min $min_w -max $max_w -f $filepath -e $max_epochs -p $norm -acc $accuracy -del $delete_results
done

# Print end date
echo "execution finished at $(date)"
