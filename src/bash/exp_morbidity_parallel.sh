#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-274  -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-1:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID/covid-env || exit
source bin/activate

# Load modules
module purge
module load CUDA/11.3.1
module load torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID/src/bash/ || exit

# Stampa degli argomenti di input

#!/usr/bin/env bashc

while getopts c:i: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        c) config_mode=${OPTARG};;
        i) id_exp=${OPTARG};;
    esac
done
echo "config_mode: $config_mode";
echo "id_exp: $id_exp";






#!/usr/bin/bash
# RUN YOUR PROGRAM
python launch_bash.py -e="$config_mode" -id="$id_exp"
# Deactivate venv
deactivate