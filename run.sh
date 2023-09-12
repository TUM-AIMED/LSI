#!/bin/sh
 
#SBATCH --job-name=indiv_privacy
#SBATCH --output=indiv_privacy-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=indiv_privacy-%A.err  # Standard error of the script
#SBATCH --time=0-1:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=20G  # Memory in GB (Don't use more than 48GB per GPU unless you absolutely need it and know what you are doing)

# load python module
ml python/anaconda3
 
# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate indiv_privacy # If this does not work, try 'source activate ptl'
 
# run the program
python mimic_experiments/compute_anything.py