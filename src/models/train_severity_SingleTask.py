import argparse
import sys
import os
import itertools


print('Python %s on %s' % (sys.version, sys.platform))
print(os.getcwd(),  ' the current working directory')
sys.path.extend('./')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import collections
import yaml
from src import mkdir, seed_all, seed_worker, get_SingleTaskModel, plot_training, evaluate, is_debug, train_severity, \
    evaluate_regression
from src.utils.utils_data import DatasetImgBX
from src.utils.utils_visualization import plot_regression
# Configuration file






def main():
    # Configuration file
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("--cfg_file", help="Number of folder", type=str)
    parser.add_argument("--model_name", help="model_name")
    parser.add_argument("--unfreeze", help="not freezed layers", default=-1)
    parser.add_argument("--id_exp", help="seed", default=1)
    args = parser.parse_args()

    # Load configuration file
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Seed everything
    seed_all(cfg['seed'])

    # Parameters
    classes = cfg['data']['classes']
    model_name = args.model_name
    steps = ['train', 'val', 'test']
    cv = cfg['data']['cv']
    fold_list = list(range(cv)) if isinstance(cv, int) else [int(value) for value in os.listdir(cfg['data']['fold_dir'])][1:]
    print(fold_list)

    validation_name = {True: 'Cross validation double stratified', False: 'Leave-one-Center-out'}
    # Data config
    data_cfg = cfg['data']['modes']['img']
    CV = '_' + str(cfg['data']['cv'])
    CLAHE = '_Clahe' if data_cfg['preprocess']['clahe'] else ''
    Filter = '_Filter3th' if data_cfg['preprocess']['filter'] else ''
    Clip = '_Clip2-98' if data_cfg['preprocess']['clip'] else ''
    Drop = f'_Drop{cfg["model"]["dropout_rate"]}'
    Batch = f'_Batch{cfg["data"]["batch_size"]}'
    LearningRate = f'_LR{cfg["trainer"]["optimizer"]["lr"]}'
    Masked = '_LungMask' if data_cfg['preprocess']['masked'] else '_Entire'
    bbox_resize = '_LungBbox' if data_cfg['preprocess']['bbox_resize'] else '_Entire' if cfg['data']['modes']['img']['bbox_resize'] else ''
    regression = '_regression-' + cfg['model']['regression_type']
    freezing = '_unfreeze_' if cfg['model']['freezing'] else ''
    warming = f'_warmup_' if cfg['trainer']['warmup_epochs'] != 0 else ''
    loss = f'_loss_{cfg["trainer"]["loss"]}' if cfg['trainer']['loss'].lower() != 'mse' else ''

    # Experiment name
    exp_name = cfg['exp_name'] + regression + CV + Batch + LearningRate + warming + loss + Drop + CLAHE + Filter + Clip + Masked + bbox_resize + freezing



    # Device
    device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
    num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
    print(device)

    # Print Report
    print('| Report experiment  |'.center(60, '-'))
    print('1) CV :', validation_name[isinstance(cfg['data']['cv'], int)])
    print('2) Model :', model_name)
    print('3) Exp name :', exp_name)
    print('4) Device :', device)
    print('5) Num workers :', num_workers)
    print('6) Classes :', classes)
    print('7) Masked :', ' Masked lung' if data_cfg['preprocess']['masked'] else ' Entire image')
    print('8) Bbox resize :', ' Lung bbox' if data_cfg['preprocess']['bbox_resize'] else ' Entire CXR')
    print('9) Dropout :', cfg['model']['dropout_rate'])
    print('10) Batch size :', cfg['data']['batch_size'])
    print('11) Learning rate :', cfg['trainer']['optimizer']['lr'])
    print('12) Clahe :', 'Applied' if data_cfg['preprocess']['clahe'] else 'Not applied')
    print('13) Filter :', 'Applied' if data_cfg['preprocess']['filter'] else 'Not applied')
    print('14) Clip :', 'Applied' if data_cfg['preprocess']['clip'] else 'Not applied')
    print(''.center(60, '-') + '\n')


    # Directories
    cfg['exp_name'] = cfg['exp_name'] + f'_{args.id_exp}'
    cfg['data']['model_dir'] = os.path.join(cfg['data']['model_dir'], cfg['exp_name'])  # folder to save trained model
    cfg['data']['report_dir'] = os.path.join(cfg['data']['report_dir'], cfg['exp_name'])
    # Files and Directories
    assert model_name in [
              "vgg11", "shufflenet_v2_x1_5", "squeezenet1_0", "squeezenet1_1", "mobilenet_v2",
              "vgg11_bn",
              "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
              "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
              "densenet121", "densenet169", "densenet161", "densenet201", "googlenet", "shufflenet_v2_x0_5",
              "shufflenet_v2_x1_0",  "resnext50_32x4d", "wide_resnet50_2"]
    model_dir = os.path.join(cfg['data']['model_dir'], exp_name, model_name) # folder to save model
    print(' ----------| Model directory: ', model_dir)


    mkdir(model_dir)
    report_dir = os.path.join(cfg['data']['report_dir'], exp_name, model_name) # folder to save results
    print(' ----------| Report directory: ', report_dir)

    mkdir(report_dir)

    plot_training_dir = os.path.join(report_dir, "training_plot")

    mkdir(plot_training_dir)
    plot_test_dir = os.path.join(report_dir, "test_plot")
    mkdir(plot_test_dir)

    # Results table
    report_file = os.path.join(report_dir, 'report_' + str(cv) + '.xlsx')
    metrics_file = os.path.join(report_dir, 'report_' + str(cv) + '_metrics.xlsx')
    report_file_temp = os.path.join(report_dir, 'report_' + str(cv) + '_temp.xlsx')
    report_metrics_temp = os.path.join(report_dir, 'report_' + str(cv) + '_metrics_temp.xlsx')


    # Results table

    rows = ['CC', 'SD', 'MSE', 'L1', 'R2']
    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'RL', 'LL', 'G']
    columns_fold = list(itertools.chain(*[[str(fold) + f' {zone}' for zone in columns] for fold in fold_list]))
    df_results = pd.DataFrame(index=rows, columns=columns_fold)




    for fold in fold_list:
        # Dir
        model_fold_dir = os.path.join(model_dir, str(fold))
        mkdir(model_fold_dir)
        plot_training_fold_dir = os.path.join(plot_training_dir, str(fold))
        mkdir(plot_training_fold_dir)
        plot_test_fold_dir = os.path.join(plot_test_dir, str(fold))
        mkdir(plot_test_fold_dir)


        # Data Loaders for SEVERITY TASK
        fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], str(fold), '%s.txt' % step), delimiter=" ") for step in steps}

        if is_debug():
            ind = [17130509736980299451 == im for im in fold_data['train']['img'].to_list()].index(True)
            fold_data['train'] = fold_data['train'][2000:2100]
            fold_data['val'] = fold_data['val'][:]
            fold_data['test'] = fold_data['test'][:]



        datasets = {
            step: DatasetImgBX(data=fold_data[step], classes=classes, cfg=cfg['data']['modes']['img'], step=step) for
            step in steps}

        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)}
        #
        idx_to_class = {v: k for k, v in datasets['train'].class_to_idx.items()}
        # Model
        if is_debug():
            for k, (inputs, labels, file_name) in enumerate(data_loaders["train"]):
                print(inputs.shape, labels.shape, file_name)



            input, _, _ = next(iter(data_loaders["train"]))
            input, _, _ = next(iter(data_loaders["val"]))
            input, _, _ = next(iter(data_loaders["test"]))
        model = get_SingleTaskModel(backbone=model_name, cfg=cfg, device=device)
        print(model)
        
        if cfg['model']['pretrained']:
            model.load_state_dict(torch.load(os.path.join(cfg['model']['pretrained'], str(fold), "model.pt"), map_location=device))

        model = model.to(device)

        # Loss function
        if cfg['trainer']['loss'].lower() == 'mse':
            criterion = nn.MSELoss().to(device)
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['optimizer']['lr'], weight_decay=cfg['trainer']['optimizer']['weight_decay'])
        # LR Scheduler

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode=cfg['trainer']['scheduler']['mode'],
                                                   patience=cfg['trainer']['scheduler']['patience'],
                                                   min_lr=float(cfg['trainer']['scheduler']['min_lr']),
                                                   factor=cfg['trainer']['scheduler']['factor'])

        # Train model

        model, history = train_severity(model=model,
                                         criterion=criterion,
                                         model_file_name=f'model_{model_name}.pt',
                                         dataloaders=data_loaders,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         num_epochs=cfg['trainer']['max_epochs'],
                                         max_epochs_stop=cfg['trainer']['early_stopping'],
                                         model_dir=model_fold_dir,
                                         device=device,
                                         regression_type=cfg['model']['regression_type'])


        # Plot Training
        plot_regression(history=history, plot_training_dir=plot_training_fold_dir)
        # Evaluate the model on all the test data
        metrics_test, loss_test = evaluate_regression(model, data_loaders['test'], criterion, idx_to_class, device, regression_type=cfg['model']['regression_type'])


        # Results inserted in the table
        for column in metrics_test.T.iterrows():
            df_results.loc[:, str(fold) + ' ' + column[0]] = column[1].values
        df_results_temp = df_results.copy(deep=True)
        for column in columns[::-1]:
            selected_cols = [str(fold_) + f' {column}' for fold_ in fold_list]
            df_results_temp.insert(0, 'std ' + column, value=df_results[selected_cols].std(1))
            df_results_temp.insert(0, 'mean ' + column, value=df_results[selected_cols].mean(1))
        df_results_temp.to_excel(report_file_temp, index=False)



    # Save Results
    results_frame = df_results.copy(deep=True)
    for column in columns[::-1]:
        selected_cols = [str(fold_) + f' {column}' for fold_ in fold_list]
        results_frame.insert(0, 'std ' + column, value=results_frame[selected_cols].std(1))
        results_frame.insert(0, 'mean ' + column, value=results_frame[selected_cols].mean(1))

    results_frame.to_excel(report_file, index=False)






if __name__ == '__main__':
    main()
