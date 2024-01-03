
import re
import sys
import os

import numpy as np
from sklearn.metrics import classification_report
from scipy.stats import wilcoxon, ttest_rel, f_oneway

print('Python %s on %s' % (sys.version, sys.platform))
print(os.getcwd(), ' the current working directory')
sys.path.extend('./')
import pandas as pd

import argparse
from imblearn.metrics import geometric_mean_score

def mkdir(path: str):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)






# Parser object for managing input arguments from command line
parser = argparse.ArgumentParser(description="Configuration File")
parser.add_argument("--report_dir", help="Report_directory", type=str, default="/mimer/NOBACKUP/groups/naiss2023-6-336/fruffini/COVID-19-MultiTask/reports")
args = parser.parse_args()

def main():






    for release_n in ['3', '2', '1']:



        for cv in ['loCo', '5']:

            # Lets create a directory where all the statistical analysis are going to be saved
            statistical_report_dir = os.path.join(args.report_dir, 'statistical_analysis')
            print(os.path.abspath(statistical_report_dir))
            mkdir(statistical_report_dir)
            release_dir = os.path.join(statistical_report_dir, 'release_{}_cv_{}'.format(release_n, cv))
            mkdir(release_dir)

            # BASELINE REPORT ANALYSIS
            AFC_cv_exp_list = os.listdir(os.path.join(args.report_dir, 'AFC', cv))
            for run in AFC_cv_exp_list:
                if f'{release_n}release' in run:
                    print(run)
                    name_experiment = run.split('task')[-1][1:]
                    image_exps_list = os.listdir(os.path.join(args.report_dir, 'AFC', cv, run))
                    pattern_T = rf'({""}).*?({re.escape("Entire")})'


                    for image_T_exp in image_exps_list:
                        match_T = re.search(pattern_T, image_T_exp)
                        if match_T:
                            # Now we have to scan the directory to find the models that have been trained
                            # and the corresponding reports
                            models_list = os.listdir(os.path.join(args.report_dir, 'AFC', cv, run, image_T_exp))
                            models_list.remove('all')
                            if len(models_list) == 0:
                                continue
                            # Create a report for all the models
                            df_final_metrics = pd.DataFrame(columns=['Model', 'Accuracy', 'G-mean', 'precision', 'recall', 'f1-score'])
                            all_models_performance = 0.0
                            number_of_samples = 0.0 # Number of samples in the test set
                            performances_by_model = {}

                            for model_name in models_list:
                                df_new_performances_calculated = {}
                                model_path = os.path.join(args.report_dir, 'AFC', cv, run, image_T_exp, model_name)

                                # Get the test prediction report
                                report_test = os.path.join(model_path, 'test_prediction')

                                # List all the directory inside the test_prediction directory
                                reports_list = os.listdir(report_test)

                                # List of result by CV fold test
                                cv_test_by_model = []

                                for cv_result in reports_list:

                                    results_by_images = os.path.join(report_test, cv_result,f'[{cv_result}]_results_by_image_{model_name}.xlsx')
                                    # Open the excel file

                                    df_result_by_image = pd.read_excel(results_by_images )
                                    df_result_by_image['CV'] = np.ones_like(df_result_by_image['Patient'].values) * int(cv_result)


                                    # Append the result to the list
                                    cv_test_by_model.append(df_result_by_image)

                                    #print(df_result_by_image)
                                    # Compute new metric

                                    g_mean = geometric_mean_score(df_result_by_image['labels'], df_result_by_image['Predicted'], average='weighted')
                                    accuracy = sum(df_result_by_image['Predicted'] == df_result_by_image['labels']) / len(df_result_by_image['Predicted'].values)


                                    classification_report_ = classification_report(df_result_by_image['labels'], df_result_by_image['Predicted'], output_dict=True)
                                    weighted_avg_report = classification_report_['weighted avg']
                                    weighted_avg_report['G-mean'] = g_mean
                                    weighted_avg_report['Accuracy'] = accuracy

                                    df_new_performances_calculated[cv_result] = weighted_avg_report
                                # Create Pandas DF
                                df_new_performances_calculated['mean'] = pd.DataFrame(df_new_performances_calculated).T.mean(axis=0)
                                df_new_performances_calculated['std'] = pd.DataFrame(df_new_performances_calculated).T.std(axis=0)
                                df_new_performances_calculated['SE'] = pd.DataFrame(df_new_performances_calculated).T.sem(axis=0)
                                df_new_performances_calculated = pd.DataFrame(df_new_performances_calculated).T
                                # Calculate STD and MEAN for Performances by CV fold



                                df_all_fold = pd.concat(cv_test_by_model)

                                computed_boolean = df_all_fold['Predicted'] == df_all_fold['labels']
                                accuracy_model = sum(computed_boolean) / len(computed_boolean.values)

                                all_models_performance += sum(computed_boolean)
                                number_of_samples += len(computed_boolean.values)

                                weighted_avg_report_cvs = classification_report(df_all_fold['labels'], df_all_fold['Predicted'], output_dict=True)['weighted avg']
                                df_final_metrics.loc[len(df_final_metrics), :] = {'Model': model_name, 'Accuracy': accuracy_model, 'G-mean': geometric_mean_score(df_all_fold['labels'],
                                                                                  df_all_fold['Predicted'], average='weighted'), **weighted_avg_report_cvs}
                                # SAVE NEW PERFORMANCES METRICS
                                df_new_performances_calculated.to_csv(os.path.join(model_path, f'{model_name}_W_perf_AFC.csv'))
                                performances_by_model[model_name] = df_new_performances_calculated

                            # COMPUTE PERFORMANCES OVERALL - ALL MODELS
                            accuracy_overall = all_models_performance / number_of_samples

                            df_final_metrics.loc[len(df_final_metrics), :] = {'Model': 'Overall', 'Accuracy': accuracy_overall}

                            df_final_metrics.to_excel(os.path.join(release_dir, f'BASELINE_performances_absolute_relative_{release_n}.xlsx'))


            # MULTI REPORT ANALYSIS
            exp_dir = os.path.join(args.report_dir, 'Multi', cv if not cv == 'loCo' else 'loCo6')
            MULTI_cv_exp_list = os.listdir(os.path.join(args.report_dir, 'Multi', cv if not cv == 'loCo' else 'loCo6'))
            for run in MULTI_cv_exp_list:
                if f'{release_n}release' in run:
                    print(run)
                    name_experiment = run.split('task')[-1][1:]
                    image_exps_list = os.listdir(os.path.join(exp_dir, run))
                    pattern_T = rf'({""}).*?({re.escape("Entire")})'


                    for image_T_exp in image_exps_list:
                        match_T = re.search(pattern_T, image_T_exp)
                        if match_T:
                            # Now we have to scan the directory to find the models that have been trained
                            # and the corresponding reports
                            image_exp_dir = os.path.join(exp_dir, run, image_T_exp)
                            models_list = os.listdir(image_exp_dir)
                            models_list.remove('all')
                            if len(models_list) == 0:
                                continue
                            if '.DS_Store' in models_list:
                                models_list.remove('.DS_Store')
                            # Create a report for all the models
                            df_final_metrics_multi = pd.DataFrame(columns=['Model', 'Accuracy', 'G-mean', 'precision', 'recall', 'f1-score'])

                            all_models_performance = 0.0
                            number_of_samples = 0.0 # Number of samples in the test set
                            performances_by_model = {}

                            for model_name in models_list:
                                df_new_performances_calculated = {}
                                model_path = os.path.join(image_exp_dir, model_name)

                                # Get the test prediction report
                                report_test = os.path.join(model_path, 'test_prediction')

                                # List all the directory inside the test_prediction directory
                                reports_list = os.listdir(report_test)


                                # List of result by CV fold test
                                cv_test_by_model = []
                                for i, cv_result in enumerate(reports_list):

                                    results_by_images = os.path.join(report_test, cv_result,f'[{cv_result}]_results_by_image_{model_name}AFC_.xlsx')
                                    # Open the excel file

                                    df_result_by_image = pd.read_excel(results_by_images )
                                    df_result_by_image['CV'] = np.ones_like(df_result_by_image['Patient'].values) * int(i)

                                    # Append the result to the list
                                    cv_test_by_model.append(df_result_by_image)
                                    #print(df_result_by_image)
                                    # Compute new metric

                                    g_mean = geometric_mean_score(df_result_by_image['labels'], df_result_by_image['Predicted'], average='weighted')
                                    accuracy = sum(df_result_by_image['Predicted'] == df_result_by_image['labels']) / len(df_result_by_image['Predicted'].values)


                                    classification_report_ = classification_report(df_result_by_image['labels'], df_result_by_image['Predicted'], output_dict=True)
                                    weighted_avg_report = classification_report_['weighted avg']
                                    weighted_avg_report['G-mean'] = g_mean
                                    weighted_avg_report['Accuracy'] = accuracy

                                    df_new_performances_calculated[cv_result] = weighted_avg_report
                                # Create Pandas DF
                                df_new_performances_calculated['mean'] = pd.DataFrame(df_new_performances_calculated).T.mean(axis=0)
                                df_new_performances_calculated['std'] = pd.DataFrame(df_new_performances_calculated).T.std(axis=0)
                                df_new_performances_calculated['SE'] = pd.DataFrame(df_new_performances_calculated).T.sem(axis=0)
                                df_new_performances_calculated = pd.DataFrame(df_new_performances_calculated).T
                                # Calculate STD and MEAN for Performances by CV fold


                                # COMPUTE PERFROMANCES OVER ALL CROSS_VALIDATION FOLDS - SINGLE MODEL
                                df_all_fold = pd.concat(cv_test_by_model)

                                computed_boolean = df_all_fold['Predicted'] == df_all_fold['labels']
                                accuracy_model = sum(computed_boolean) / len(computed_boolean.values)

                                all_models_performance += sum(computed_boolean)
                                number_of_samples += len(computed_boolean.values)
                                # Classification Report All CVs
                                classification_report_ = classification_report(df_all_fold['labels'], df_all_fold['Predicted'], output_dict=True)
                                weighted_avg_report_cvs = classification_report_['weighted avg']

                                df_final_metrics_multi.loc[len(df_final_metrics_multi), :] = {'Model': model_name, 'Accuracy': accuracy_model, 'G-mean': geometric_mean_score(df_all_fold['labels'], df_all_fold['Predicted'], average='weighted'),
                                **weighted_avg_report_cvs}

                                # SAVE NEW PERFORMANCES METRICS
                                df_new_performances_calculated.to_csv(os.path.join(model_path, f'{model_name}_W_perf_AFC.csv'))
                                performances_by_model[model_name] = df_new_performances_calculated

                            # COMPUTE PERFORMANCES OVERALL - ALL MODELS
                            accuracy_overall = all_models_performance / number_of_samples

                            df_final_metrics_multi.loc[len(df_final_metrics_multi), :] = {'Model': 'Overall', 'Accuracy': accuracy_overall}

                            df_final_metrics_multi.to_excel(os.path.join(release_dir, f'MULTI_{release_n}release_performances_RUN{run}.xlsx'))


                            # ---- STATISTICAL ANALYSIS ----
                            # 1. Paired t-test
                            # 2. F-test
                            # 3. Wilcoxon signed-rank test
                            # 4. Binomial test
                            report_statistical_ANALYSIS = os.path.join(release_dir, f'statistical_analysis_{release_n}release_RUN{run}.xlsx')
                            dict_statistical_analysis = {}
                            for metric_to_valuate in ['Accuracy', 'G-mean', 'precision', 'recall', 'f1-score']:
                                baseline_acc = df_final_metrics.loc[:(len(df_final_metrics)-2)][metric_to_valuate].to_list()
                                multi_acc = df_final_metrics_multi.loc[:(len(df_final_metrics_multi)-2)][metric_to_valuate].to_list()
                                paired_t_stat, paired_p_value = ttest_rel(baseline_acc, multi_acc, alternative='less')
                                print('-' * 50)
                                print(f'Metric: {metric_to_valuate}'.center(50, '-'))
                                print('-'*50)


                                print(f'Paired t-test: {paired_t_stat}, p-value: {paired_p_value}')
                                wilcoxon_stat, wilcoxon_p_value = wilcoxon(baseline_acc, multi_acc, alternative = 'less')


                                print(f'Wilcoxon signed-rank test: {wilcoxon_stat}, p-value: {wilcoxon_p_value}')
                                f_statistic, p_value = f_oneway(baseline_acc, multi_acc)

                                print(f'F-test: {f_statistic}, p-value: {p_value}')

                                dict_statistical_analysis[metric_to_valuate] = {'paired_t_stat': paired_t_stat, 'paired_p_value': paired_p_value, 'wilcoxon_stat': wilcoxon_stat,
                                                                                'wilcoxon_p_value': wilcoxon_p_value, 'f_statistic': f_statistic, 'f_p_value': p_value}


                            pd.DataFrame.from_dict(dict_statistical_analysis, orient='index').to_excel(report_statistical_ANALYSIS)


































if __name__ == '__main__':
    main()