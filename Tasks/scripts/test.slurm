#!/bin/sh

# My email address
#SBATCH --mail-user=oiliossatos@gmail.com
#SBATCH --mail-type=end

# How much time my job must run (min 1 minute)
#SBATCH --time=00:01:00

# How much memory (RAM) my job will use (min 512MB)
#SBATCH --mem=512MB

# How many nodes I'm going to use
#SBATCH --nodes=1

# How many cores/cpu must be reserved for my job.
#SBATCH --cpus-per-task=1

# The name of the queue/partition I want to use.
#SBATCH --partition slurm-cluster

# Name of the job
#SBATCH --job-name=dummy-test

# Where do the output logs go?
#SBATCH --output=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/test.%J.out
#SBATCH --error=/home/oliossat/Documents/Semester-Project/Tasks/scripts/outputs/test.%J.err

# Navigate to job location in directory
cd ~/Documents/Semester-Project/Tasks/

# Print the name of the machine is running on
echo "executed on $HOSTNAME"

# Print the start date
echo "execution started on $(date)"

# Run the program with srun which is used to submit a job for execution in real time
# Note: srun is blocking. You cannot write other commands until it is finished.
srun python test.py

# Print end date
echo "execution finished at $(date)"
