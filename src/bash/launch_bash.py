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
parser.add_argument("-e", "--experiment_config", help="Number of folder", type=str, choices=['5', 'L', '10'],
                    default='5')
parser.add_argument("-k", "--modality_kind", help="Type of modality", type=str,
                    choices=['morbidity', 'severity', 'both'], default='morbidity')

parser.add_argument("--model_names", help="model_name", default=
[	
    "densenet201",
    "densenet121",
    "densenet161",
    "googlenet",
    "mobilenet_v2", 
    "squeezenet1_0",
    "squeezenet1_1",
    "densenet121",
    "vgg11_bn",
    "densenet169",
    "wide_resnet50_2",
    "vgg11",
    "squeezenet1_0",
    "squeezenet1_1",
    "alexnet",
    "vgg11_bn",
    "vgg16",
    "vgg16_bn",
    "vgg13_bn",
    "vgg13",
    "resnet18",
    "resnet34",
    "resnet50",

    "resnet101",
    "resnet152",
    "googlenet",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "resnext50_32x4d",
    "wide_resnet50_2",
])

"""    "resnet18",
    "resnet34",
    "resnet50",
"""



parser.add_argument("-id", "--exp_id", help="not freezed layers", default=1, type=int)
args = parser.parse_args()

config_selector = {
    'morbidity':
    {
        '5': '../../configs/bash_experiments/experiment_setups_morbidity_5.json',
        'L': '../../configs/bash_experiments/experiment_setups_morbidity_loCo.json',
    },
    'severity':
        {
            '5': '../../configs/bash_experiments/experiment_setups_severity_5.json',
            'L': '../../configs/bash_experiments/experiment_setups_severity_loCo.json'
        }
}


if __name__ == "__main__":

    # Load the experiment list
    file_config = config_selector[args.modality_kind][args.experiment_config]
    print(file_config)
    with open(file_config, 'r') as data_file:
        json_data = data_file.read()

    experiment_list = json.loads(json_data)
    print(experiment_list)
    processes = []  # List to store the subprocess instances


    for i, exp_config in enumerate(experiment_list):

        args.model_names = eval(exp_config['models']) if exp_config['models'] != 'all' else args.model_names

        print('exp ', exp_config, 'models', args.model_names)

        
        for model in args.model_names:


            # Imposta la variabile d'ambiente con il dizionario
            os.environ["config_dir"] = "configs/{}/{}/{}".format(str(exp_config['fold']), args.modality_kind,
                                                                       exp_config['config_file'])
            os.environ["model_name"] = str(model)
            os.environ["id_exp"] = str(args.exp_id)
            print("modality", args.modality_kind,
                  "model", model, "\n",
                  "id", args.exp_id,"\n",
                  "fold", exp_config['fold'],"\n",
                  "config", exp_config['config_file'])
            launch_slurm_job('train_severity.sh' if args.modality_kind == "severity" else 'train_morbidity.sh',
                             os.environ)



