#!/bin/bash
#SBATCH --account=def-jijup
#SBATCH --job-name=train_facade_test
#SBATCH --gres=gpu:1                # Change this to suit your case. It is set to 1 now
#SBATCH --cpus-per-task=6           # change this to suit your case. It is set to 6 now
#SBATCH --mem=64G                   # Change this to suit your case. It is set to 64G now.You can do a test run and use seff <jobid> to get an estimate
#SBATCH --time=0-00:50              # Change this to suit your case. It is set to 50 minutes now
#SBATCH --output=%N-%j.out

# Purge  to remove any existing modules to avoid conflicts
module purge
# Load modules in correct order. For example these are the ones I needed for SqueezeSegV3 training.
# Check the ones you need to load or install. I.e Scipy-stack contains numpy. scipy, pandas etc. no need to install them.
module load StdEnv/2023 StdEnv/2020 gcc cuda opencv/4.5.5 python/3.10 scipy-stack/2022a

# Create fresh virtual environment with correct Python
virtualenv --no-download --system-site-packages my_env_new

# Activate the virtual environment
source /home/abdiaziz/SqueezeSegV3_New/my_env_new/bin/activate

# Verify Python version
python --version

# Upgrade pip first - Didn't had an issue without upgrading too.
pip install --upgrade pip

# Install requirements. Use --no-index to avoid using PyPI. Local wheels are optimized for the environment.
pip install --no-index -r /home/abdiaziz/SqueezeSegV3_New/requirements.txt

# Set PYTHONPATH
export PYTHONPATH="/home/abdiaziz/SqueezeSegV3_New/src:$PYTHONPATH"

# Run training
python train_facade.py --dataset /home/abdiaziz/scratch/processed # Store the dataset in the scratch directory for faster IO