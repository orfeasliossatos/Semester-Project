#!/bin/sh

#SBATCH --mail-user=oiliossatos@gmail.com
#SBATCH --mail-type=end
#SBATCH --time=06:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition slurm-gpu
#SBATCH --job-name=dim_vs_size
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
min_w=4
max_w=14
max_epochs=100
norm=2
accuracy=0.8
delete_results=True
filepath="results/week3/p${norm}_acc{accuracy}_e{max_epochs}.pkl"

# Run python script
props=(
    0.2
    0.4
    0.6
    0.8
    1
)


for i in "${props[@]}"; do
    srun python dim_vs_tr_size.py -actv "Quad" -arch "ParamFairCNN" -prop {proportion} -min $min_w -max $max_w -f $filepath -e $max_epochs -p $norm -acc $accuracy -del $delete_results
done

# Print end date
echo "execution finished at $(date)"
