import argparse
import subprocess
import json
import os


def launch_slurm_job(script_path, env):
    command = ['sbatch', script_path]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        job_id = extract_job_id(stdout)
        print(stdout.decode("utf-8"))
        print(f'Successfully submitted SLURM job with ID {job_id}')
    else:
        print(f'Error submitting SLURM job: {stderr.decode("utf-8")}')


def extract_job_id(sbatch_output):
    output_lines = sbatch_output.decode("utf-8").split('\n')
    # The last line of the sbatch output contains the job ID
    job_id_line = output_lines[-2]
    job_id = job_id_line.split()[-1]
    return job_id





# Configuration file
parser = argparse.ArgumentParser(description="Configuration File")
parser.add_argument("--experiment_config", help="Number of folder", type=str,
                    default='../../configs/bash_experiments/experiment_setups_morbidity_5.json')

parser.add_argument("--model_names", help="model_name", default=
    [
     'alexnet',
     'resnet18',
     'resnet34',
     'resnet50',
     'resnet101',
     'resnet152',
     'densenet121',
     'densenet169',
     'densenet161',
     'densenet201',
     'shufflenet_v2_x0_5',
     'shufflenet_v2_x1_0',
     'mobilenet_v2'
     ])
"""

     'resnext50_32x4d',
     'wide_resnet50_2',
     'mnasnet0_5',
     'mnasnet1_0',
     'vgg11',
     'vgg11_bn',
     'vgg13',
     'vgg13_bn',
     'vgg16',
     'vgg16_bn',
     'vgg19',
     'vgg19_bn'
"""
parser.add_argument("--unfreeze", help="not freezed layers", default=-1)
args = parser.parse_args()






if __name__ == "__main__":



    with open(args.experiment_config, 'r') as data_file:
        json_data = data_file.read()

    experiment_list = json.loads(json_data)

    processes = []  # List to store the subprocess instances
    for model in args.model_names:

        for i, exp_config in enumerate(experiment_list):
            # Imposta la variabile d'ambiente con il dizionario
            os.environ["config_dir"] = "configs/{}/morbidity/{}".format(str(exp_config['fold']), exp_config['config_file'])
            os.environ["model_name"] = str(model)
            launch_slurm_job('train_morbidity.sh', os.environ)
