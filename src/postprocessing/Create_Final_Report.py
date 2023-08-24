import os
import pandas as pd


def reportCreation(modality = 'Morbidity'):

    modality_set = {"Morbidity": 1, "Severity":0 }


    if modality_set[modality]:
        CV = 5

        Experiment_folders = [
        f"reports/AFC/{str(CV)}/{modality.lower().strip()}_singletask_1/{modality.lower().strip()}_singletask_{str(CV)}_Batch64_LR0.001_Drop0.25_LungMask_LungBbox",
        f"reports/AFC/{str(CV)}/{modality.lower().strip()}_singletask_1/{modality.lower().strip()}_singletask_{str(CV)}_Batch64_LR0.001_Drop0.25_Entire_LungBbox",
        ]

        for Experiment_folder, type_exp in zip(Experiment_folders, ['LungMask', 'Entire']):

            Models_folders = os.listdir(Experiment_folder)

            df_resume_experiments = pd.DataFrame(index=Models_folders,
                                                 columns=['mean ACC', 'std ACC', 'mean ACC MILD', 'std ACC MILD',
                                                          'mean ACC SEVERE', 'std ACC SEVERE', 'mean AUC', 'std AUC',
                                                          'mean Precision', 'std Precision', 'mean Recall',
                                                          'std Recall', 'mean F1', 'std F1'])

            print('Experiment_folder: ', Experiment_folder)

            for Model_folder in Models_folders:

                print('Model_folder: ', Model_folder)

                model_report_dir = os.path.join(Experiment_folder, Model_folder)

                file_report = os.path.join(model_report_dir, f'report_{CV}.xlsx')
                metrics_report = os.path.join(model_report_dir, f'report_{CV}_metrics.xlsx')

                if os.path.exists(file_report):
                    file_result_cv = pd.read_excel(file_report, index_col=0)
                    file_result_cv_metrics = pd.read_excel(metrics_report, index_col=0)

                    extraction_file_results = file_result_cv.iloc[0, :6]
                    extraction_file_results_metrics = file_result_cv_metrics.iloc[0, :8]

                    concatenated_results = pd.concat([extraction_file_results, extraction_file_results_metrics],
                                                     axis=0).to_frame().T

                    df_resume_experiments.loc[Model_folder] = concatenated_results.values[0]

            df_resume_experiments.dropna(inplace=True)
            df_resume_experiments.to_excel(
                os.path.join(Experiment_folder, f'models_report_{CV}_resume_exp_{type_exp}.xlsx'))

            print('here')



    else:
        CV = 'loCo'

        Experiment_folders = [
        f"reports/BX/{str(CV)}/{modality.lower().strip()}_singletask_2/{modality.lower().strip()}_singletask_regression-area_{str(CV)}_Batch64_LR0.001_Drop0.25_LungMask_LungBbox",
        f"reports/BX/{str(CV)}/{modality.lower().strip()}_singletask_2/{modality.lower().strip()}_singletask_regression-area_{str(CV)}_Batch64_LR0.001_Drop0.25_Entire_LungBbox",
        ]

        df_CC_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])
        df_SD_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])
        df_MSE_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])
        df_L1_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])
        df_R2_results = pd.DataFrame(columns=['mean A', 'std A', 'mean B', 'std B', 'mean C', 'std C', 'mean D', 'std D', 'mean E', 'std E', 'mean F', 'std F', 'mean RL', 'std RL', 'mean LL', 'std LL', 'mean G', 'std G'])







        for Experiment_folder, type_exp in zip(Experiment_folders, ['LungMask', 'Entire']):

            Models_folders = os.listdir(Experiment_folder)

            df_resume_experiments = pd.DataFrame(index=Models_folders,
                                                 columns=['mean ACC', 'std ACC', 'mean ACC MILD', 'std ACC MILD',
                                                          'mean ACC SEVERE', 'std ACC SEVERE', 'mean AUC', 'std AUC',
                                                          'mean Precision', 'std Precision', 'mean Recall',
                                                          'std Recall', 'mean F1', 'std F1'])

            print('Experiment_folder: ', Experiment_folder)

            for Model_folder in Models_folders:

                print('Model_folder: ', Model_folder)

                model_report_dir = os.path.join(Experiment_folder, Model_folder)







                file_report = os.path.join(model_report_dir, f'report_{CV}.xlsx')

                if os.path.exists(file_report):
                    rows = ['CC', 'SD', 'MSE', 'L1', 'R2']
                    file_result_cv = pd.read_excel(file_report)
                    file_result_cv.index = ['CC', 'SD', 'MSE', 'L1', 'R2']

                    result_cc = file_result_cv.iloc[0, : 18]
                    df_CC_results.loc[Model_folder] = result_cc



                    result_sd = file_result_cv.iloc[1, : 18]
                    df_SD_results.loc[Model_folder] = result_sd




                    result_mse = file_result_cv.iloc[2, : 18]
                    df_MSE_results.loc[Model_folder] = result_mse




                    result_l1 = file_result_cv.iloc[3, : 18]
                    df_L1_results.loc[Model_folder] = result_l1





                    result_r2 = file_result_cv.iloc[4, : 18]
                    df_R2_results.loc[Model_folder] = result_r2
                    df_R2_results.dropna(inplace=True)

            df_CC_results.to_excel(
                os.path.join(Experiment_folder, f'models_CC_report_{CV}_resume_exp_{type_exp}.xlsx'))
            df_SD_results.to_excel(
                os.path.join(Experiment_folder, f'models_SD_report_{CV}_resume_exp_{type_exp}.xlsx'))
            df_MSE_results.to_excel(
                os.path.join(Experiment_folder, f'models_MSE_report_{CV}_resume_exp_{type_exp}.xlsx'))
            df_L1_results.to_excel(
                os.path.join(Experiment_folder, f'models_L1_report_{CV}_resume_exp_{type_exp}.xlsx'))
            df_R2_results.to_excel(
                os.path.join(Experiment_folder, f'models_R2_report_{CV}_resume_exp_{type_exp}.xlsx'))




            print('here')


if __name__ == "__main__":
    reportCreation()