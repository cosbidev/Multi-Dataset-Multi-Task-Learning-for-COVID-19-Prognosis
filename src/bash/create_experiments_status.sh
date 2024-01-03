#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-493  -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-00:10:00
# Output files
#SBATCH --error=EXP_REPORT_job_%J.err
#SBATCH --output=EXP_REPORT_out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

echo %J
# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID/covid-env || exit
source bin/activate


module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
while getopts n: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        n) release_number=${OPTARG};;
    esac
done

echo "release_number: $release_number";
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID/src/postprocessing/ || exit
#!/usr/bin/bash
# RUN YOUR PROGRAM
python create_experiment_status.py -n $release_number