#!/bin/sh

#SBATCH --mail-user=oiliossatos@gmail.com
#SBATCH --time=8:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition slurm-gpu
#SBATCH --job-name=size_vs_acc
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
min_mag=2
max_mag=5
splits=10
max_epochs=100
norm=2
width=8
delete_results=True
filepath="results/week1/p${norm}_l${width}_e${max_epochs}.pkl"

# Architectures and activation functions
archs=(
	"DumbCNN"
	"FCNN"
)

actvs=(
	"Quad"
	"ReLU"
)

# Run python script

for i in "${archs[@]}"; do
	for j in "${actvs[@]}"; do
		srun python tr_size_vs_te_acc.py -actv "$j" -arch "$i" -min $min_mag -max $max_mag -f $filepath -s $splits -e $max_epochs -p $norm -l $width -del $delete_results
	done
done
# Print end date
echo "execution finished at $(date)"

