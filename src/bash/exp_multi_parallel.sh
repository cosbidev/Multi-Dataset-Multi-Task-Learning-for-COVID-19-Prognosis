#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-274  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 0-00:10:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID/covid-env || exit
source bin/activate

module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-image/0.19.3-foss-2022a
module load scikit-learn/1.1.2-foss-2022a
module load  OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib
# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID/src/bash/ || exit

# Stampa degli argomenti di input

#!/usr/bin/env bashc

while getopts c:i:h flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        c) config_mode=${OPTARG};;
        i) id_exp=${OPTARG};;
        h) checkpoint='-c';;
    esac
done
echo "config_mode: $config_mode";
echo "id_exp: $id_exp";
echo "checkpoint: $checkpoint";






#!/usr/bin/bash
# RUN YOUR PROGRAM
python launch_bash.py -e="$config_mode" -id="$id_exp" -k multi $checkpoint
# Deactivate venv
deactivate