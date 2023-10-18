#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-274  -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 1-1:00:00

#SBATCH --error=job_%J_%c.err
#SBATCH --output=out_%J_%i.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruffin02@outlook.it
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



