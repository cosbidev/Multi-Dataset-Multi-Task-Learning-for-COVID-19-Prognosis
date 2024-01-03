#!/usr/bin/env bash
module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-image/0.19.3-foss-2022a
module load scikit-learn/1.1.2-foss-2022a
module load  OpenCV/4.6.0-foss-2022a-CUDA-11.7.0-contrib
# Executes the code
cd /mimer/NOBACKUP/groups/snic2022-5-277/fruffini/ItaChinaCOVID19/ProgettoAnno1/MultiObjective_BRIXIA-AIforCOVID || exit

while getopts m:i:s:d: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        m) modality=${OPTARG};;
        i) id_exp=${OPTARG};;
        s) structure=${OPTARG};;
        d) root_dir=${OPTARG};;
    esac
done
echo "modality: $modality";
echo "id_exp: $id_exp";
echo "structure: $structure";
echo "root_dir: $root_dir";

#d
# Train HERE YOU RUN YOUR PROGRAM
python src/postprocessing/Create_Final_Report.py --modality ${modality} --name_exp=${id_exp} --structure="${structure}" --root=${root_dir}