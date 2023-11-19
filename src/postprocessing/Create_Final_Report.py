import glob
import os
import shutil

import numpy as np
import pandas as pd
import argparse


def reportCreation():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("--modality", "-m", help="Modalities", choices=['morbidity','severity', 'multi'], type=str, required=True)
    parser.add_argument("--name_exp", "-exp", help="model_name", default='MISE')
    parser.add_argument("--structure", "-s", help="for Multi", default='', choices=['serial', 'parallel', 'none'])
    parser.add_argument("--version", "-v", help="Version", default='all', choices=['all'])
    args = parser.parse_args()

    # Parameters
    modality = args.modality
    directory_id = args.name_exp
    multi_structure = args.structure
    v_ = args.version
    modality_set = {"morbidity": 1, "severity": 0, "multi": 2}
    versions = {'R':(['resnet18', 'resnet34', 'resnet50'], 'resnets18-34-50'), 'all' :([], 'all')}
    save_combined = {'loCo': {'Entire': None, 'Masked': None}, '5': {'Entire': None, 'Masked': None}} if modality_set[modality.lower()] == 1 else\
        {'loCo': {'serial': None, 'parallel': None},
         '5': {'serial': None, 'parallel': None},
         'loCo6': {'serial': None, 'parallel': None},
         'loCo18': {'serial': None, 'parallel': None}}
    if modality_set[modality.lower()] == 1 or modality_set[modality.lower()] == 2:
        fold_cv = ['loCo', '5'] if modality_set[modality.lower()] == 1 else ['5', 'loCo6', 'loCo18']
        for CV in fold_cv:
            ext_exp = 'AFC' if modality_set[modality.lower()] == 1 else 'Multi'
            task_ = 'singletask' if modality_set[modality.lower()] == 1 else 'multitask'
            type_model = 'morbidity' if modality_set[modality.lower()] == 1 else multi_structure
            experiment_folder = f"reports/{ext_exp}/{str(CV)}/{type_model}_{task_}_{directory_id}"
            print(experiment_folder, 'existe: ', os.path.exists(experiment_folder))
            if not os.path.exists(experiment_folder):
                continue

            Experiment_folders = [dir for dir in os.scandir(experiment_folder) if os.path.isdir(dir)]

            for Experiment_folder in Experiment_folders:

                Models_folders = os.listdir(Experiment_folder)

                df_resume_experiments = pd.DataFrame(index=Models_folders,
                                                     columns=['Accuracy_mean', 'Accuracy_std','Precision_mean','Precision_std','Recall_mean','Recall_std','F1 Score_mean','F1 Score_std','ROC AUC Score_mean','ROC AUC Score_std','MILD_accuracy_mean','MILD_accuracy_std','SEVERE_accuracy_mean','SEVERE_accuracy_std'])

                for Model_folder in Models_folders:
                    version = versions[v_]
                    if version[0].__len__() > 0:
                        if Model_folder not in version[0] or Model_folder == 'all':
                            continue
                    # Masked or Unmasked experiment
                    type_of_run = 'Masked' if 'LungMask' in Experiment_folder.name else 'Entire'
                    model_report_dir = os.path.join(Experiment_folder, Model_folder)
                    if '.DS' in model_report_dir:
                        continue
                    os.listdir(model_report_dir)
                    file_report = os.path.join(model_report_dir, f'[all]_test_results_{Model_folder}')
                    if modality_set[modality.lower()] == 2:

                        if 'Curriculum' in directory_id:
                            print( 'Curriculum SAVED BEST')
                            number = 5 if CV == 'loCo' else 4
                            file_report = os.path.join(model_report_dir, f'test_prediction/{number}/[{number}]_results_prediction_{Model_folder}bestABS_AFC_.xlsx')
                    else:
                        file_report = file_report + f'.xlsx'
                    if os.path.exists(file_report):
                        file_result_cv = pd.read_excel(file_report, index_col=0)

                        mean_row = file_result_cv.loc['mean', :].to_dict()
                        std_row = file_result_cv.loc['std', :].to_dict()
                        combined_results = {}
                        for (key_mean, mean_metric), (key_std, std_metric) in zip(mean_row.items(), std_row.items()):
                            combined_results[key_mean + '_mean'] = np.round(mean_metric, 2)
                            combined_results[key_mean + '_std'] = np.round(std_metric, 2)

                        df_resume_experiments.loc[Model_folder] = combined_results

                df_resume_experiments.dropna(inplace=True)
                print('Number of models Trained: '
                      'Model\'s list: ', df_resume_experiments.index.to_list())
                df_resume_experiments = df_resume_experiments.sort_values(by=['Accuracy_mean'], ascending=False)

                save_combined[CV][type_of_run] = df_resume_experiments

                save_name = Experiment_folder.name.split('singletask')[-1]

                # Report folder
                rep_version = os.path.join(Experiment_folder.path, f'{v_}')

                if not os.path.exists(rep_version):
                    os.mkdir(rep_version)
                df_resume_experiments.to_csv(
                    os.path.join(rep_version, f'Morbidity_models_report_{CV}_resume_exp_{save_name}_{version[1]}_{directory_id}.csv'))

    if modality_set[modality.lower()] == 0 or modality_set[modality.lower()] == 2:
        fold_cv = ['loCo', '5'] if modality_set[modality.lower()] == 0 else ['5', 'loCo6', 'loCo18']
        for CV in fold_cv:
            ext_exp = 'BX' if modality_set[modality.lower()] == 0 else 'Multi'
            task_ = 'singletask' if modality_set[modality.lower()] == 0 else 'multitask'
            type_model = 'severity' if modality_set[modality.lower()] == 0 else multi_structure
            experiment_folder = f"reports/{ext_exp}/{str(CV)}/{type_model}_{task_}_{directory_id}"
            print(experiment_folder, 'exist: ', os.path.exists(experiment_folder))
            if not os.path.exists(experiment_folder):
                continue
            Experiment_folders = [dir for dir in os.scandir(experiment_folder) if os.path.isdir(dir)]
            for Experiment_folder in Experiment_folders:
                Models_folders = os.listdir(Experiment_folder)
                df_resume_experiments = pd.DataFrame(index=Models_folders,
                                                     columns=['Accuracy L1_mean','Accuracy L1_std','Accuracy Exp_mean','Accuracy Exp_std','Accuracy Squared_mean','Accuracy Squared_std',
                                                              'Acc_G_mean','Acc_G_std','Acc_LR_mean','Acc_LR_std','Acc_LL_mean','Acc_LL_std','LL_L1_mean','LL_L1_std','RL_L1_mean','RL_L1_std'
                                                         ,'G_L1_mean','G_L1_std','LL_CC_mean','LL_CC_std','RL_CC_mean','RL_CC_std','G_CC_mean','G_CC_std'])

                for Model_folder in Models_folders:
                    version = versions[v_]
                    if version[0].__len__() > 0:
                        if Model_folder not in version[0] or Model_folder == 'all':
                            shutil.rmtree(os.path.join(Experiment_folder, Model_folder))
                            continue
                    # Masked or Unmasked experiment
                    type_of_run = 'Masked' if 'LungMask' in Experiment_folder.name else 'Entire'
                    model_report_dir = os.path.join(Experiment_folder, Model_folder)
                    if '.DS' in model_report_dir:
                        continue
                    os.listdir(model_report_dir)
                    file_report = os.path.join(model_report_dir, f'[all]_test_results_{Model_folder}')

                    if modality_set[modality.lower()] == 2:
                        file_report = file_report + f'_BX.xlsx'
                    else:
                        file_report = file_report + f'.xlsx'
                    if os.path.exists(file_report):
                        file_result_cv = pd.read_excel(file_report, index_col=0)
                        mean_row = file_result_cv.loc['mean', :].to_dict()
                        std_row = file_result_cv.loc['std', :].to_dict()
                        combined_results = {}
                        for (key_mean, mean_metric), (key_std, std_metric) in zip(mean_row.items(), std_row.items()):
                            combined_results[key_mean + '_mean'] = np.round(mean_metric, 2)
                            combined_results[key_mean + '_std'] = np.round(std_metric, 2)
                        df_resume_experiments.loc[Model_folder] = pd.Series(combined_results)
                df_resume_experiments = df_resume_experiments.sort_values(by=['Acc_G_mean'], ascending=False)
                save_combined[CV][type_of_run] = df_resume_experiments
                save_name = Experiment_folder.name.split('singletask')[-1]
                # Report folder
                rep_version = os.path.join(Experiment_folder.path, f'{v_}')
                if not os.path.exists(rep_version):
                    os.mkdir(rep_version)




                df_resume_experiments.dropna(inplace=True)
                print('Number of models Trained: '
                      'Model\'s list: ', df_resume_experiments.index.to_list())
                df_resume_experiments.to_csv(
                    os.path.join(rep_version, f'Severity_models_report_{CV}_resume_exp_{save_name}_{version[1]}_{directory_id}.csv'))


                print('Experiment_Report: ', os.path.join(rep_version, f'Severity_models_report_{CV}_resume_exp_{save_name}_{version[1]}_{directory_id}.csv'))


if __name__ == "__main__":
    reportCreation()