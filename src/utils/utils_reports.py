import os



def compute_report_metrics(final_report_folds,
                           metrics_report,
                           fold,
                           results_by_patient,
                           model_name,
                           report_path,
                           save_temp=True,
                           classes_report=None,
                           classes=None,
                           optional_Dataset=''
                           ):
    # FINAL RESULTS:
    # 1) Save all the prediction for each image in the val set
    optional_Dataset += '_' if optional_Dataset != '' else ''
    results_by_patient.to_excel(os.path.join(report_path, f'[{str(fold)}]_results_by_image_{model_name}{optional_Dataset}.xlsx'), index=False)

    # 2) Add metrics computed by classes and by fold in the final_report_folds
    final_report_folds.loc['fold_' + str(fold), :] = metrics_report
    if classes_report is not None and classes is not None:
        for class_ in classes:
            final_report_folds.loc['fold_' + str(fold), class_ + '_accuracy'] = classes_report[classes_report['class'] == class_]['top1'].item()

    # 3) Add mean and std of the metrics computed in the result_metrics_test selected by the fold
    final_report_folds.loc['mean', :] = final_report_folds[[True if 'fold' in row[0] else False for row in final_report_folds.iterrows()]].mean()
    final_report_folds.loc['std', :] = final_report_folds[[True if 'fold' in row[0] else False for row in final_report_folds.iterrows()]].std()
    final_report_folds.sort_index(ascending=True, inplace=True)

    # 4) save temp file
    if save_temp:
        final_report_folds.to_excel(os.path.join(report_path, f'[{str(fold)}]_results_prediction_{model_name}{optional_Dataset}.xlsx'))
    return final_report_folds
