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
module load scikit-image/0.19.3-foss-2022a
module load scikit-learn/1.1.2-foss-2022a
module load  OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib


cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID || exit
#!/usr/bin/bash
# RUN YOUR PROGRAM
python src/postprocessing/statistical_analysis/compute_statistical_scores.py