#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-274  -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-12:00:00
# Output files


#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it

echo  "cfg : $config_dir", "with model: $model_name"


# Activate venv
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID/covid-env || exit
source bin/activate


# Load modules
#module purge
#module load CUDA/11.3.1
#module load torchvision/0.12.0-foss-2021a-PyTorch-1.11.0-CUDA-11.3.1
module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-image/0.19.3-foss-2022a
module load scikit-learn/1.1.2-foss-2022a
module load  OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib
# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID || exit

config=$config_dir
model=$model_name
id_exp=$id_exp
#!/usr/bin/bash
# Train HERE YOU RUN YOUR PROGRAM
python src/models/train_MultiObjectiveModel.py --model_name ${model} --cfg_file=${config} --id_exp=${id_exp}
# Deactivate venv
deactivate