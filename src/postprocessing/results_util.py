import glob
import os
import re

import numpy as np
import pandas as pd

def transform_number(num):
    # Define the criteria for identification
    if num < 0.4:
        # If the number is less than 1, multiply it by 100
        transformed_num = round(num * 100, 3)
    else:
        # If the number is greater than or equal to 1, leave it as is
        transformed_num = round(num, 3)
    return transformed_num
def load_experiments_results(paths: list, experiments: list, setups_cv: list, image_types: list, head_modes: list, metrics: list, models_names: list):
    path = paths[0]
    all_list = {}
    list_all_model_missing = {}
    data = []
    # Experiments directory:
    for experiment in experiments:

        if '2release' in experiment:
            release_class = 'AFC12'
        elif '3release' in experiment:
            release_class = 'AFC123'
        elif '1release' in experiment:
            release_class = 'AFC1'



        model_list = []
        if "BASELINE" in experiment:
            # AiForCovid directory BASELINE
            name_dir = 'AFC'
            # Name Mapping for the plot release
            path_exp = os.path.join(path, name_dir)
            old_name = experiment
            experiment_new = experiment
            # Name Mapping for the cv strategy
            mapper_cv = {'CV5': '5', 'loCo': 'loCo'}
            inverse_mapper_cv  =  {v: k for k, v in mapper_cv.items()}
            # No Head for Baseline
            head_modes_selection = ['']

        else:
            # MultiTask-Objective directory
            name_dir = 'Multi'
            path_exp = os.path.join(path, name_dir)
            mapper_cv = {'CV5': '5', 'loCo': 'loCo6'}
            inverse_mapper_cv  = {v: k for k, v in mapper_cv.items()}
            head_modes_selection = head_modes.copy()
            head_modes_selection.remove('baseline')

            old_name = experiment
            experiment_new = experiment
        for setup_cv in setups_cv:
            # Map CV name:
            setup_cv = mapper_cv[setup_cv]
            # Map Exp Run Name:

            path_exp_cv_runs = os.listdir(os.path.join(path_exp, setup_cv))
            if len(path_exp_cv_runs) == 0:
                continue
            # Get all the runs for the experiment
            for head_ in head_modes_selection:
                # Define Pattern
                if head_:
                    pattern = rf'({re.escape(head_)}).*?({re.escape(experiment_new)})$'
                else:
                    pattern = rf'({re.escape(experiment_new)})'

                for run in path_exp_cv_runs:
                    match = re.search(pattern, run)
                    if match:
                        path_exp_cv = os.path.join(path_exp, setup_cv, match.string)
                    else:
                        continue
                    # Get all the runs for the experiment and the cv strategy and the head mode (if any) and the image type (Mask/Entire)
                    for image_type in image_types:
                        pattern_T = rf'({""}).*?({re.escape(image_type)})'

                        path_exp_cv_imagesT = os.listdir(path_exp_cv)

                        for image_T_exp in path_exp_cv_imagesT:
                            match_T = re.search(pattern_T, image_T_exp)

                            if match_T:
                                path_exp_cv_images = os.path.join(path_exp_cv, image_T_exp)

                                pattern_report = r'.*Morbidity.*\.csv'

                                file_catcher = [filename for filename in os.listdir(os.path.join(path_exp_cv_images, 'all')) if re.search(pattern_report,
                                                                                                                        filename)]
                                print(experiment)
                                if len(file_catcher) == 0:
                                    continue
                                else:
                                    all_report = pd.read_csv(
                                        os.path.join(path_exp_cv_images, 'all', file_catcher[0])).set_index('Unnamed: 0')
                                    all_list[old_name + '_' + setup_cv + '_' + 'head' + '_' + image_type] = all_report
                                    for model in models_names:
                                        if model in all_report.index:
                                            model_row = all_report.loc[model, :]

                                        else:
                                            model_list.append(model)
                                            continue
                                        for measure in ['mean', 'std']:

                                            measure_row = model_row[[measure in col for col in model_row.index]]

                                            for metric in metrics:
                                                mapper_metric = {'F1 Score': 'F1_score', 'Accuracy': 'Accuracy', 'SEVERE_accuracy': 'Accuracy-Severe', 'MILD_accuracy': 'Accuracy-Mild',
                                                                 'ROC AUC Score': 'ROC-AUC'}
                                                inverse_mapper_metric = {v: k for k, v in mapper_metric.items()}
                                                head_local = 'baseline' if head_ == '' else head_

                                                if metric == 'ROC-AUC':
                                                    pass
                                                measure_row_metric = measure_row[inverse_mapper_metric[metric] + '_' + measure]
                                                measure_row_metric= transform_number(measure_row_metric)
                                                data.append({'Image Type': image_type, 'Model': model, 'Experiment': old_name,
                                                             'Category': measure, 'Cross Val Strategy': inverse_mapper_cv[setup_cv],
                                                             'Head Mode': head_local, 'Value': measure_row_metric, 'Metric': metric, 'Release': release_class})
                                    list_all_model_missing[experiment + '_' + image_type + '_' + old_name + '_' + inverse_mapper_cv[setup_cv] + '_' + head_local] = list(np.unique(
                                        model_list))

    return pd.DataFrame(data)

