import argparse
import glob
import itertools
import sys
import os

print('Python %s on %s' % (sys.version, sys.platform))
print(os.getcwd(), ' the current working directory')
sys.path.extend('./')
from easydict import EasyDict

from src.utils.utils_visualization import plot_training_multi

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import collections
import yaml
from src import mkdir, seed_all, MultiTaskDataset, seed_worker, get_MultiTaskModel, plot_training, train_morbidity, evaluate_morbidity, is_debug, train_MultiTask, IdentityMultiHeadLoss, \
    evaluate_multi_task, Logger
from src.utils import utils_data

# Configuration file


def main():
    # Configuration file
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("--cfg_file", help="Number of folder", type=str)
    parser.add_argument("--model_name", help="model_name")
    parser.add_argument("--id_exp", help="seed", default=1)
    args = parser.parse_args()

    # Load configuration file
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Seed everything
    seed_all(cfg['seed'])
    cfg = EasyDict(cfg)
    # Parameters
    # Datasets configs
    morbidity_cfg = cfg.data.modes.morbidity
    severity_cfg = cfg.data.modes.severity

    # BACKBONE
    model_name = args.model_name

    steps = ['train', 'val', 'test']
    cv = cfg['data']['cv']

    # Data config
    data_cfg = cfg['data']
    CV = '_' + str(cfg['data']['cv'])
    Batch = f'_Batch{cfg["data"]["batch_size"]}'

    # Preprocessing config
    CLAHE = '_Clahe' if data_cfg['preprocess']['clahe'] else ''
    Filter = '_Filter3th' if data_cfg['preprocess']['filter'] else ''
    Clip = '_Clip2-98' if data_cfg['preprocess']['clip'] else ''
    Masked = '_LungMask' if data_cfg['preprocess']['masked'] else '_Entire'


    # Model Config
    model_cfg = cfg['model']
    Drop = f'_Drop{model_cfg["dropout_rate"]}'
    freezing = '_unfreeze' if not model_cfg['freezing'] else ''
    regression = f'_regression_{model_cfg["regression_type"]}' if model_cfg['regression_type'] else ''
    pretrained = '_pretrained_BX' if model_cfg['pretrained'] else ''

    # Trainer Config:
    trainer_cfg = cfg['trainer']
    warming = f'_warmup_' if trainer_cfg['warmup_epochs'] != 0 else ''
    LearningRate = f'_LR{trainer_cfg["optimizer"]["lr"]}'


    # Experiment name
    exp_name = cfg['exp_name'] + CV + Batch + regression + pretrained + LearningRate + warming + Drop  + CLAHE + Filter + Clip + Masked + freezing
    print(' ----------| Experiment name: ', exp_name)

    # Device
    device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
    num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
    print(device)
    # Directories
    cfg['exp_name'] = cfg['exp_name'] + f'_{args.id_exp}'
    cfg['data']['model_dir'] = os.path.join(cfg['data']['model_dir'], cfg['exp_name'])  # folder to save trained model
    cfg['data']['report_dir'] = os.path.join(cfg['data']['report_dir'], cfg['exp_name'])
    # Files and Directories
    assert model_name in ["resnet18", "resnet34", "resnet50"]

    model_dir = os.path.join(cfg['data']['model_dir'], exp_name, model_name)  # folder to save model
    print(' ----------| Model directory: ', model_dir)

    mkdir(model_dir)
    report_dir = os.path.join(cfg['data']['report_dir'], exp_name, model_name)  # folder to save results
    print(' ----------| Report directory: ', report_dir)

    mkdir(report_dir)
    logger = Logger(file_name=os.path.join(report_dir, f'log_print_out_{model_name}.txt'), file_mode="w", should_flush=True)
    with open(os.path.join(report_dir, f'config_{model_name}.yaml'), 'w') as file:
        documents = yaml.dump(cfg, file)
    plot_training_dir = os.path.join(report_dir, "training_plot")

    plot_training_dir = os.path.join(report_dir, "training_plot")

    mkdir(plot_training_dir)
    plot_test_dir = os.path.join(report_dir, "test_plot")
    mkdir(plot_test_dir)

    # Create Fold Array for MORBIDITY TASK and SEVERITY TASK (same folds)
    cv_option = cfg['data']['cv']


    fold_grid, fold_list = utils_data.create_combined_folds(cv_option=cv_option, morbidity_cfg=morbidity_cfg, severity_cfg=severity_cfg)
    if is_debug():
        fold_grid = {fold: fold_grid[fold] for fold in fold_list[:2]}
        fold_list = list(fold_grid.keys())
    # Results table
    report_file = os.path.join(report_dir, 'report_' + str(cv) + '.xlsx')
    metrics_file = os.path.join(report_dir, 'report_' + str(cv) + '_metrics.xlsx')

    # Results table (S)
    rows_S = ['CC', 'SD', 'MSE', 'L1', 'R2']
    columns_S = ['A', 'B', 'C', 'D', 'E', 'F', 'RL', 'LL', 'G']
    columns_fold_S = list(itertools.chain(*[[str(fold) + f' {zone}' for zone in columns_S] for fold in fold_list]))
    df_results_S = pd.DataFrame(index=rows_S, columns=columns_fold_S)

    # Results table (M)
    classes_morbidity = morbidity_cfg['classes']
    results_frame_morbidity = {}
    acc_cols = []
    acc_cat_cols = collections.defaultdict(lambda: [])
    for fold in fold_list:
        acc_col = str(fold) + " ACC"
        acc_cols.append(acc_col)
        results_frame_morbidity[acc_col] = []
        for cat in classes_morbidity:
            cat_col = str(fold) + " ACC " + cat
            acc_cat_cols[cat].append(cat_col)
            results_frame_morbidity[cat_col] = []
    acc_cat_cols = dict(acc_cat_cols)

    results_metrics_morbidity = {}
    f1_cols, acc_cols_test, auc_cols, recall_cols, precision_cols = [], [], [], [], []
    for fold in fold_list:
        acc_col_test = str(fold) + " ACC test "
        f1_col = str(fold) + " F1"
        precision_col = str(fold) + " precision"
        recall_col = str(fold) + " recall"
        auc_col = str(fold) + " auc"

        acc_cols_test.append(acc_col_test)
        f1_cols.append(f1_col)
        auc_cols.append(precision_col)
        recall_cols.append(recall_col)
        precision_cols.append(auc_col)
        results_metrics_morbidity[acc_col_test] = []
        results_metrics_morbidity[f1_col] = []
        results_metrics_morbidity[precision_col] = []
        results_metrics_morbidity[recall_col] = []
        results_metrics_morbidity[auc_col] = []



    for fold, fold_data in fold_grid.items():

        string_fold = '-----------| Fold ' + str(fold) + ' |----------'
        print(''.center(len(string_fold), '-'))
        print(''.center(len(string_fold), '-'))
        print(string_fold)
        print(''.center(len(string_fold), '-'))
        print(''.center(len(string_fold), '-'))
        # Dir
        model_fold_dir = os.path.join(model_dir, str(fold))
        mkdir(model_fold_dir)
        plot_training_fold_dir = os.path.join(plot_training_dir, str(fold))
        mkdir(plot_training_fold_dir)
        plot_test_fold_dir = os.path.join(plot_test_dir, str(fold))
        mkdir(plot_test_fold_dir)

        # Data

        if is_debug():
            fold_data['train'] = fold_data['train'][740:1040:3]
            fold_data['val'] = fold_data['val'][113:413:2]
            fold_data['test'] = fold_data['test'][113:413:2]



        # ------------------- DATA -------------------
        datasets = {step: MultiTaskDataset(data=fold_data[step], cfg_morbidity=morbidity_cfg, cfg_severity=severity_cfg, step=step, cfg=cfg) for step in steps}

        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)}
        idx_to_class = datasets['train'].m_idx_to_class

        # ------------------- MODEL -------------------
        model = get_MultiTaskModel(backbone=model_name, cfg=cfg, device=device)


        if cfg['model']['pretrained']:
            ids_experiment_to_load = cfg['model']['pretrained']

            task = {'AFC':'morbidity', 'BX': 'severity'}




            selector_exp_folder = {
                5: lambda name_dataset, id_: f'./models/{name_dataset}/5/{task[name_dataset]}_singletask_{id_}/',
                'loCo': lambda name_dataset, id_: f'./models/{name_dataset}/loCo/{task[name_dataset]}_singletask_{id_}/',

            }
            id_AFC = ids_experiment_to_load[0]
            id_BX = ids_experiment_to_load[1]
            cv_option_loading = 'loCo' if 'loco' in str(cv_option).lower() else cv_option
            exp_folder_morbidity = selector_exp_folder[cv_option_loading](name_dataset='AFC', id_=id_AFC)
            exp_folder_severtity =  selector_exp_folder[cv_option_loading](name_dataset='BX', id_=id_BX)

            exp_single_task_selector = lambda folder_singletask: glob.glob(os.path.join(folder_singletask, '*Masked_LungBbox')) if cfg['data']['preprocess']['masked'] \
                                        else glob.glob(os.path.join(folder_singletask, '*Entire_LungBbox'))


            exp_folder_AFC = exp_single_task_selector(exp_folder_morbidity)[0]
            exp_folder_BX = exp_single_task_selector(exp_folder_severtity)[0]

            model_weights_path = lambda dataset_name, cv_option, fold: os.path.join(exp_folder_BX,  model_name, str(fold)) if cv_option != 5 or cv_option != 'loCo6' else os.path.join(
                exp_folder_BX,  model_name, str(fold))

            model_dir_weights_S = os.path.join(exp_folder_BX,  model_name, str(fold))
            model_dir_weights_M = os.path.join(exp_folder_AFC,  model_name, str(fold))



            files_S = list(os.scandir(model_dir_weights_S))
            file_M = list(os.scandir(model_dir_weights_M))
            if len(files_S) > 1:
                times = [os.path.getatime(path) for path in files_S]
                file_S = files_S[times.index(min(times))]

            else:
                file_S = files_S[0]
            model_Severity = torch.load(file_S.path, map_location=device)
            severity_params = model_Severity.named_parameters()


            # Morbidity
            id_BX = ids_experiment_to_load[1]
            experiment_dir = f'models/AFC/5/morbidity_singletask_{id_BX}/'
            pretrained_Morbidity_folder = 'morbidity_singletask_5_Batch64_LR0.001_Drop0.25_Entire_LungBbox'
            pretrained_folder_Morbidity = os.path.join(experiment_dir, pretrained_Morbidity_folder)
            model_dir_weights_M = os.path.join(pretrained_folder_Morbidity,  model_name, str(fold))
            files_M = list(os.scandir(model_dir_weights_M))

            if len(files_M) > 1:
                times = [os.path.getatime(path) for path in files_M]
                file_M = files_M[times.index(min(times))]

            else:
                file_M = files_M[0]
            model_Morbidity = torch.load(file_M.path, map_location=device)
            morbidity_params = model_Morbidity.named_parameters()

            # Load weights for the parameters as the mean of the two models pretrained
            Beta = cfg['model']['beta']
            model.load_backbone_average_weights(morbidity_params, severity_params, Beta = Beta)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model.activate_Head_training_module()
            model = nn.DataParallel(model, [0,1])

        model = model.to(device)


        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['optimizer']['lr'], weight_decay=cfg['trainer']['optimizer']['weight_decay'])
        # LR Scheduler

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode=cfg['trainer']['scheduler']['mode'],
                                                   patience=cfg['trainer']['scheduler']['patience'],
                                                   min_lr=float(cfg['trainer']['scheduler']['min_lr']),
                                                   factor=cfg['trainer']['scheduler']['factor'])

        # Multi Head Identity Loss Handling
        criterion = IdentityMultiHeadLoss(cfg=cfg).to(device)
        # Train model

        model, history = train_MultiTask(model=model,
                                         model_file_name=f'{model_cfg.head}_{model_name}',
                                         dataloaders=data_loaders,
                                         cfg=cfg,
                                         criterion=criterion,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         num_epochs=cfg['trainer']['max_epochs'],
                                         max_epochs_stop=cfg['trainer']['early_stopping'],
                                         model_dir=model_fold_dir,
                                         device=device)

        # Plot Training
        plot_training_multi(history, plot_training_dir)
        # Evaluate the model on all the test data
        results_morbidity, metrics_morbidity, loss_test_severity, metrics_severity = evaluate_multi_task(
                                                                                                        cfg=cfg,
                                                                                                        model=model,
                                                                                                        test_loader=data_loaders['test'],
                                                                                                        criterion=criterion,
                                                                                                        device=device,
                                                                                                        idx_to_class_AFC=idx_to_class,
                                                                                                        topk=(1,)
                                                                                                        )
        acc = metrics_morbidity['Accuracy']

        # Test model
        print(' ----------| (M) Results Test model ')
        print(metrics_morbidity)
        print(' ----------| (S) Results Test model ')
        print(metrics_severity)
        print(loss_test_severity['Loss_S'].item())

        # Update report (M)
        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------

        results_frame_morbidity[str(fold) + " ACC"].append(acc)

        for cat in classes_morbidity:
            results_frame_morbidity[str(fold) + " ACC " + str(cat)].append(results_morbidity.loc[results_morbidity["class"] == cat]["top1"].item())

        results_metrics_morbidity[str(fold) + " F1"].append(metrics_morbidity['F1 Score'])
        results_metrics_morbidity[str(fold) + " ACC test "].append(acc)
        results_metrics_morbidity[str(fold) + " precision"].append(metrics_morbidity['Precision'])
        results_metrics_morbidity[str(fold) + " recall"].append(metrics_morbidity['Recall'])
        results_metrics_morbidity[str(fold) + " auc"].append(metrics_morbidity['ROC AUC Score'])

        # Save temporary Results
        results_frame_temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_frame_morbidity.items()]))
        for cat in classes_morbidity[::-1]:
            results_frame_temp.insert(loc=0, column='std ACC ' + cat,
                                      value=results_frame_temp[acc_cat_cols[cat]].std(axis=1))
            results_frame_temp.insert(loc=0, column='mean ACC ' + cat,
                                      value=results_frame_temp[acc_cat_cols[cat]].mean(axis=1))
        results_frame_temp.insert(loc=0, column='std ACC', value=results_frame_temp[acc_cols].std(axis=1))
        results_frame_temp.insert(loc=0, column='mean ACC', value=results_frame_temp[acc_cols].mean(axis=1))
        results_frame_temp.insert(loc=0, column='model', value=model_name)

        # Save temporary Metrics
        results_metrics_morbidity_temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_metrics_morbidity.items()]))
        results_metrics_morbidity_temp.insert(loc=0, column='std F1', value=results_metrics_morbidity_temp[f1_cols].std(axis=1))
        results_metrics_morbidity_temp.insert(loc=0, column='mean F1', value=results_metrics_morbidity_temp[f1_cols].mean(axis=1))
        results_metrics_morbidity_temp.insert(loc=0, column='std Recall', value=results_metrics_morbidity_temp[recall_cols].std(axis=1))
        results_metrics_morbidity_temp.insert(loc=0, column='mean Recall', value=results_metrics_morbidity_temp[recall_cols].mean(axis=1))
        results_metrics_morbidity_temp.insert(loc=0, column='std Precision', value=results_metrics_morbidity_temp[precision_cols].std(axis=1))
        results_metrics_morbidity_temp.insert(loc=0, column='mean Precision', value=results_metrics_morbidity_temp[precision_cols].mean(axis=1))
        results_metrics_morbidity_temp.insert(loc=0, column='std AUC', value=results_metrics_morbidity_temp[auc_cols].std(axis=1))
        results_metrics_morbidity_temp.insert(loc=0, column='mean AUC', value=results_metrics_morbidity_temp[auc_cols].mean(axis=1))
        results_metrics_morbidity_temp.insert(loc=0, column='model', value=model_name)
        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------

        # Results inserted in the table
        for column in metrics_severity.T.iterrows():
            df_results_S.loc[:, str(fold) + ' ' + column[0]] = column[1].values
        df_results_temp_S = df_results_S.copy(deep=True)
        for column in columns_S[::-1]:
            selected_cols = [str(fold_) + f' {column}' for fold_ in fold_list]
            df_results_temp_S.insert(0, 'std ' + column, value=df_results_S[selected_cols].std(1))
            df_results_temp_S.insert(0, 'mean ' + column, value=df_results_S[selected_cols].mean(1))

    with pd.ExcelWriter(report_file) as writer_results:
        # Save Results (S)
        results_frame_S = df_results_S.copy(deep=True)
        for column in columns_S[::-1]:
            selected_cols = [str(fold_) + f' {column}' for fold_ in fold_list]
            results_frame_S.insert(0, 'std ' + column, value=results_frame_S[selected_cols].std(1))
            results_frame_S.insert(0, 'mean ' + column, value=results_frame_S[selected_cols].mean(1))

        results_frame_S.to_excel(writer_results, sheet_name='Severity')

        # Save Results (M)

        results_frame_morbidity = pd.DataFrame.from_dict(results_frame_morbidity)
        for cat in classes_morbidity[::-1]:
            results_frame_morbidity.insert(loc=0, column='std ACC ' + cat, value=results_frame_morbidity[acc_cat_cols[cat]].std(axis=1))
            results_frame_morbidity.insert(loc=0, column='mean ACC ' + cat, value=results_frame_morbidity[acc_cat_cols[cat]].mean(axis=1))
        results_frame_morbidity.insert(loc=0, column='std ACC', value=results_frame_morbidity[acc_cols].std(axis=1))
        results_frame_morbidity.insert(loc=0, column='mean ACC', value=results_frame_morbidity[acc_cols].mean(axis=1))
        results_frame_morbidity.insert(loc=0, column='model', value=model_name)

        results_frame_morbidity.to_excel(writer_results, index=False, sheet_name='Morbidity')

    with pd.ExcelWriter(metrics_file) as writer_metrics:
        metrics_frame_morbidity = pd.DataFrame.from_dict(results_metrics_morbidity)
        metrics_frame_morbidity.insert(loc=0, column='std F1', value=metrics_frame_morbidity[f1_cols].std(axis=1))
        metrics_frame_morbidity.insert(loc=0, column='mean F1', value=metrics_frame_morbidity[f1_cols].mean(axis=1))
        metrics_frame_morbidity.insert(loc=0, column='std Recall', value=metrics_frame_morbidity[recall_cols].std(axis=1))
        metrics_frame_morbidity.insert(loc=0, column='mean Recall', value=metrics_frame_morbidity[recall_cols].mean(axis=1))
        metrics_frame_morbidity.insert(loc=0, column='std Precision', value=metrics_frame_morbidity[precision_cols].std(axis=1))
        metrics_frame_morbidity.insert(loc=0, column='mean Precision', value=metrics_frame_morbidity[precision_cols].mean(axis=1))
        metrics_frame_morbidity.insert(loc=0, column='std AUC', value=metrics_frame_morbidity[auc_cols].std(axis=1))
        metrics_frame_morbidity.insert(loc=0, column='mean AUC', value=metrics_frame_morbidity[auc_cols].mean(axis=1))
        metrics_frame_morbidity.insert(loc=0, column='model', value=model_name)

        metrics_frame_morbidity.to_excel(writer_metrics, index=False, sheet_name='Morbidity')


if __name__ == '__main__':
    main()
