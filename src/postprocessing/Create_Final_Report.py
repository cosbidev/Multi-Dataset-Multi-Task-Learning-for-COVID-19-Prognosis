import os
import pandas as pd


def reportCreation():


    CV = 5

    Experiment_folders = [
        "reports/AFC/5/morbidity_singletask_1/morbidity_singletask_5_Batch64_LR0.001_Drop0.2_LungMask_LungBbox",
        "reports/AFC/5/morbidity_singletask_1/morbidity_singletask_5_Batch64_LR0.001_Drop0.25_Entire_LungBbox",
    ]





    for Experiment_folder, type_exp in zip(Experiment_folders, ['Masked', 'Entire']):

        Models_folders = os.listdir(Experiment_folder)


        df_resume_experiments = pd.DataFrame(index=Models_folders, columns=['mean ACC', 'std ACC', 'mean ACC MILD', 'std ACC MILD', 'mean ACC SEVERE', 'std ACC SEVERE', 'mean AUC', 'std AUC', 'mean Precision', 'std Precision', 'mean Recall', 'std Recall', 'mean F1', 'std F1'])



        print('Experiment_folder: ', Experiment_folder)


        for Model_folder in Models_folders:

            print('Model_folder: ', Model_folder)

            model_report_dir = os.path.join(Experiment_folder, Model_folder)

            file_report = os.path.join(model_report_dir, f'report_{CV}.xlsx')
            metrics_report = os.path.join(model_report_dir, f'report_{CV}_metrics.xlsx')



            if os.path.exists(file_report):
                file_result_cv = pd.read_excel(file_report, index_col=0)
                file_result_cv_metrics = pd.read_excel(metrics_report, index_col=0)


                extraction_file_results = file_result_cv.iloc[0,:6]
                extraction_file_results_metrics = file_result_cv_metrics.iloc[0,:8]

                concatenated_results = pd.concat([extraction_file_results, extraction_file_results_metrics], axis=0).to_frame().T

                df_resume_experiments.loc[Model_folder] = concatenated_results.values[0]




        df_resume_experiments.dropna(inplace=True)
        df_resume_experiments.to_excel(os.path.join(Experiment_folder, f'models_report_{CV}_resume_exp_{type_exp}.xlsx'))


        print('here')














if __name__ == "__main__":
    reportCreation()