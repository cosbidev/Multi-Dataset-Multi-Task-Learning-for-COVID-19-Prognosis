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
parser.add_argument("-e", "--experiment_config", help="Number of folder", type=str,
                    default='5')
parser.add_argument("-k", "--modality_kind", help="Type of modality", type=str,
                    choices=['morbidity', 'severity', 'multi', 'multi_cl'], default='morbidity')

parser.add_argument("--model_names", help="model_name", default=
['googlenet'])

parser.add_argument("-id", "--exp_id", help="not freezed layers")
parser.add_argument("-c", "--checkpoint", help="Number of folder", action='store_true')
args = parser.parse_args()

config_selector = {
    'morbidity':
        {
            '5': '../../configs/bash_experiments/experiment_setups_morbidity_5.json',
            '10': '../../configs/bash_experiments/experiment_setups_morbidity_10.json',
            'L': '../../configs/bash_experiments/experiment_setups_morbidity_loCo.json',
        },
    'severity':
        {
            '5': '../../configs/bash_experiments/experiment_setups_severity_5.json',
            'L': '../../configs/bash_experiments/experiment_setups_severity_loCo.json',
            'E': '../../configs/bash_experiments/experiment_setups_morbidity_E.json'
        },
    'multi':
        {
            '5': '../../configs/bash_experiments/experiment_setups_multitask_parallel_5.json',
            'L6': '../../configs/bash_experiments/experiment_setups_multitask_parallel_loCo_6.json',
            'L18': '../../configs/bash_experiments/experiment_setups_multitask_parallel_loCo_18.json',
        },
    'multi_cl':
        {
            'P5': '../../configs/bash_experiments/experiment_setups_multitask_parallel_5.json',
            'PL6': '../../configs/bash_experiments/experiment_setups_multitask_parallel_loCo_6.json',
            'PL18': '../../configs/bash_experiments/experiment_setups_multitask_parallel_loCo_18.json',
            'S5': '../../configs/bash_experiments/experiment_setups_multitask_serial_5.json',
            'SL6': '../../configs/bash_experiments/experiment_setups_multitask_serial_loCo_6.json',
            'SL18': '../../configs/bash_experiments/experiment_setups_multitask_serial_loCo_18.json',
        }
}

if __name__ == "__main__":

    file_config = config_selector[args.modality_kind][args.experiment_config]
    print(file_config)
    with open(file_config, 'r') as data_file:
        json_data = data_file.read()

    experiment_list = json.loads(json_data)
    print(experiment_list)
    processes = []  # List to store the subprocess instances

    for i, exp_config in enumerate(experiment_list):
        # pick model
        if exp_config['models'] == 'diff':
            args.model_names = [
                "densenet121_CXR",
                "densenet121",
                "efficientnet_lite0",
                "efficientnet_b0",
                "efficientnet_es",
                "efficientnet_es_pruned",
                "efficientnet_b1_pruned",
                "densenet201",
                "densenet161",
                "densenet169",
                "mobilenet_v2",
                "resnet101",
                "resnext50_32x4d",
                "wide_resnet50_2",
            ]
        elif exp_config['models'] == 'easy':
            args.model_names = \
            ["googlenet",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet50_ChestX-ray14",
            "resnet50_ChexPert",
            "resnet50_ImageNet_ChestX-ray14",
            "resnet50_ImageNet_ChexPert",
            "shufflenet_v2_x0_5",
            "shufflenet_v2_x1_0",
            ]
        elif exp_config['models'] == 'all':
            args.model_names = \
            ["googlenet",
             "resnet18",
             "resnet34",
             "resnet50",
             "resnet50_ChestX-ray14",
             "resnet50_ChexPert",
             "resnet50_ImageNet_ChestX-ray14",
             "resnet50_ImageNet_ChexPert",
             "shufflenet_v2_x0_5",
             "shufflenet_v2_x1_0",
             "densenet121_CXR",
             "densenet121",
             "efficientnet_lite0",
             "efficientnet_b0",
             "efficientnet_es",
             "efficientnet_es_pruned",
             "efficientnet_b1_pruned",
             "densenet201",
             "densenet161",
             "densenet169",
             "mobilenet_v2",
             "resnet101",
             "resnext50_32x4d",
             "wide_resnet50_2",
            ]

        else:
            args.model_names = eval(exp_config['models'])
        print('exp ', exp_config, 'models', args.model_names)

        for model in args.model_names:


            # Imposta la variabile d'ambiente con il dizionario
            os.environ["config_dir"] = "configs/{}/{}/{}".format(str(exp_config['fold']), args.modality_kind if args.modality_kind != 'multi_cl' else 'multi',exp_config['config_file'])
            if args.checkpoint:
                os.environ["checkpoint"] = "-c"
            os.environ["structure"] = exp_config['structure']
            os.environ["model_name"] = str(model)
            os.environ["id_exp"] = str(args.exp_id)
            os.environ["modality"] = str(args.modality_kind)
            if exp_config['release']:
                os.environ["release"] = str(exp_config['release'])
            print("modality", args.modality_kind,
                  "model", model, "\n",
                  "id", args.exp_id, "\n",
                  "fold", exp_config['fold'], "\n",
                  "config", exp_config['config_file'], "\n",
                  "checkpoint", args.checkpoint, "\n",
                  "structure", exp_config['structure'], "\n")
            bash_file = {'severity': 'train_severity.sh',
                         'morbidity': 'train_morbidity.sh',
                         'multi': 'train_multi.sh',
                         'multi_cl': 'train_multi_curriculum.sh'}
            launch_slurm_job(bash_file[args.modality_kind],
                             os.environ)