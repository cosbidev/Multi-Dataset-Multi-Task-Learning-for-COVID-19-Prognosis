#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277  -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:1
#SBATCH -t 0-06:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.tortora@unicampus.it

# Load modules

# Stampa degli argomenti di input
echo "Experiment number $i , with arguments: $exp"

module load CUDA/11.6.0
module load Python/3.9.6-GCCcore-11.2.0

# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/mtortora/comer/comer-venv || exit
source bin/activate

# Executes the code

cd /mimer/NOBACKUP/groups/snic2022-5-277/mtortora/comer || exit

# RUN YOUR PROGRAM
python main_bash.py

# Deactivate venv
deactivate



