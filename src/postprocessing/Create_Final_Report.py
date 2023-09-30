import os
import pandas as pd


def reportCreation(modality = 'Morbidity'):

    modality_set = {"morbidity": 1, "severity":0 , "multi": 2 }
    directory_id = 'baseline'

    v_ = 'all'

    versions = {'R':(['resnet18', 'resnet34', 'resnet50'], 'resnets18-34-50'), 'all' :([], 'all')}

    if modality_set[modality.lower()] == 1:
        for CV in ['loCo', '5']:

            experiment_folder = f"reports/AFC/{str(CV)}/{modality.lower().strip()}_singletask_{directory_id}"
            if not os.path.exists(experiment_folder):
                continue

            Experiment_folders = [dir for dir in os.scandir(experiment_folder) if os.path.isdir(dir)]

            for Experiment_folder in Experiment_folders:

                Models_folders = os.listdir(Experiment_folder)

                df_resume_experiments = pd.DataFrame(index=Models_folders,
                                                     columns=['Accuracy_mean', 'Accuracy_std','Precision_mean','Precision_std','Recall_mean','Recall_std','F1 Score_mean','F1 Score_std','ROC AUC Score_mean','ROC AUC Score_std','MILD_accuracy_mean','MILD_accuracy_std','SEVERE_accuracy_mean','SEVERE_accuracy_std'])

                print('Experiment_folder: ', Experiment_folder)

                for Model_folder in Models_folders:
                    version = versions[v_]
                    if version[0].__len__() > 0:
                        if Model_folder not in version[0] or Model_folder is 'all':
                            continue
                    print('Model_folder: ', Model_folder)

                    model_report_dir = os.path.join(Experiment_folder, Model_folder)
                    if '.DS' in model_report_dir:
                        continue
                    os.listdir(model_report_dir)
                    file_report = os.path.join(model_report_dir, f'[all]_test_results_{Model_folder}.xlsx')

                    if os.path.exists(file_report):
                        file_result_cv = pd.read_excel(file_report, index_col=0)

                        mean_row = file_result_cv.loc['mean', :].to_dict()
                        std_row = file_result_cv.loc['std', :].to_dict()
                        combined_results = {}
                        for (key_mean, mean_metric), (key_std, std_metric) in zip(mean_row.items(), std_row.items()):
                            print(mean_metric, std_metric)
                            combined_results[key_mean + '_mean'] = mean_metric
                            combined_results[key_mean + '_std'] = std_metric

                        df_resume_experiments.loc[Model_folder] = combined_results

                df_resume_experiments.dropna(inplace=True)
                df_resume_experiments = df_resume_experiments.sort_values(by=['Accuracy_mean'], ascending=False)

                save_name = Experiment_folder.name.split('singletask')[-1]

                # Report folder
                rep_version = os.path.join(f'reports/AFC/{str(CV)}/{modality.lower().strip()}_singletask_{directory_id}', Experiment_folder.name, f'{v_}')

                if not os.path.exists(rep_version):
                    os.mkdir(rep_version)
                df_resume_experiments.to_excel(
                    os.path.join(rep_version, f'models_report_{CV}_resume_exp_{save_name}_{version[1]}.xlsx'))




    elif modality_set[modality.lower()] == 0:
        for CV in ['loCo', '5']:

            Experiment_folders = [
            f"reports/BX/{str(CV)}/{modality.lower().strip()}_singletask_{directory_id}/{modality.lower().strip()}_singletask_regression-area_{str(CV)}_Batch64_LR0.001_Drop0.25_LungMask_LungBbox",
            f"reports/BX/{str(CV)}/{modality.lower().strip()}_singletask_{directory_id}/{modality.lower().strip()}_singletask_regression-area_{str(CV)}_Batch64_LR0.001_Drop0.25_Entire_LungBbox",
            ]

            df_CC_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])
            df_SD_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])
            df_MSE_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])
            df_L1_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])
            df_R2_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])




            experiment_folder = f"reports/BX/{str(CV)}/{modality.lower().strip()}_singletask_{directory_id}"
            if not os.path.exists(experiment_folder):
                continue
            Experiment_folders = [dir for dir in os.scandir(experiment_folder) if os.path.isdir(dir)]

            for Experiment_folder in Experiment_folders:

                Models_folders = os.listdir(Experiment_folder)

                df_resume_experiments = pd.DataFrame(index=Models_folders,
                                                     columns=['mean ACC', 'std ACC', 'mean ACC MILD', 'std ACC MILD',
                                                              'mean ACC SEVERE', 'std ACC SEVERE', 'mean AUC', 'std AUC',
                                                              'mean Precision', 'std Precision', 'mean Recall',
                                                              'std Recall', 'mean F1', 'std F1'])

                print('Experiment_folder: ', Experiment_folder)

                for Model_folder in Models_folders:

                    version = versions[v_]
                    if version[0].__len__() > 0:
                        if Model_folder not in version[0]:
                            continue

                    print('Model_folder: ', Model_folder)

                    model_report_dir = os.path.join(Experiment_folder, Model_folder)


                    file_report = os.path.join(model_report_dir, f'report_{CV}.xlsx')

                    if os.path.exists(file_report):
                        rows = ['CC', 'SD', 'MSE', 'L1', 'R2']
                        file_result_cv = pd.read_excel(file_report)
                        file_result_cv.index = ['CC', 'SD', 'MSE', 'L1', 'R2']

                        result_cc = file_result_cv.iloc[0, : 18]
                        df_CC_results.loc[Model_folder] = result_cc
                        df_CC_results.sort_values(by=['mean G'], ascending=False, inplace=True)


                        result_sd = file_result_cv.iloc[1, : 18]
                        df_SD_results.loc[Model_folder] = result_sd
                        df_SD_results.sort_values(by=['mean G'], ascending=True, inplace=True)



                        result_mse = file_result_cv.iloc[2, : 18]
                        df_MSE_results.loc[Model_folder] = result_mse
                        df_MSE_results.sort_values(by=['mean G'], ascending=True, inplace=True)



                        result_l1 = file_result_cv.iloc[3, : 18]
                        df_L1_results.loc[Model_folder] = result_l1
                        df_L1_results.sort_values(by=['mean G'], ascending=True, inplace=True)


                        result_r2 = file_result_cv.iloc[4, : 18]
                        df_R2_results.loc[Model_folder] = result_r2
                        df_R2_results.sort_values(by=['mean G'], ascending=True, inplace=True)
                        df_R2_results.dropna(inplace=True)


                save_name = Experiment_folder.name.split('singletask')[-1]

                df_resume_experiments.to_excel(
                    os.path.join(f'reports/BX/{str(CV)}/{modality.lower().strip()}_singletask_{directory_id}', f'models_report_{CV}_resume_exp_{save_name}_{version[1]}.xlsx'))



                df_CC_results.to_excel(
                    os.path.join(f'reports/BX/{CV}/{modality.lower().strip()}_singletask_{directory_id}', f'models_CC_report_{CV}_resume_exp_{save_name}_{version[1]}.xlsx'))
                df_SD_results.to_excel(
                    os.path.join(f'reports/BX/{CV}/{modality.lower().strip()}_singletask_{directory_id}', f'models_SD_report_{CV}_resume_exp_{save_name}_{version[1]}.xlsx'))
                df_MSE_results.to_excel(
                    os.path.join(f'reports/BX/{CV}/{modality.lower().strip()}_singletask_{directory_id}', f'models_MSE_report_{CV}_resume_exp_{save_name}_{version[1]}.xlsx'))
                df_L1_results.to_excel(
                    os.path.join(f'reports/BX/{CV}/{modality.lower().strip()}_singletask_{directory_id}', f'models_L1_report_{CV}_resume_exp_{save_name}_{version[1]}.xlsx'))
                df_R2_results.to_excel(
                    os.path.join(f'reports/BX/{CV}/{modality.lower().strip()}_singletask_{directory_id}', f'models_R2_report_{CV}_resume_exp_{save_name}_{version[1]}.xlsx'))

    elif modality_set[modality] == 2:
        for CV in ['loCo', '5']:



            print('here')


if __name__ == "__main__":
    reportCreation()