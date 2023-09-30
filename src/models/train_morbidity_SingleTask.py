import argparse
import glob
import shutil
import sys
import os

import numpy as np

print('Python %s on %s' % (sys.version, sys.platform))
print(os.getcwd(), ' the current working directory')
sys.path.extend('./')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import yaml
from src import mkdir, seed_all, DatasetImgAFC, seed_worker, get_SingleTaskModel, plot_training, \
    train_morbidity, evaluate_morbidity, is_debug, Logger, compute_report_metrics


# Configuration file


def main():
    # Configuration file
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("--cfg_file", help="Number of folder", type=str)
    parser.add_argument("--model_name", help="model_name")
    parser.add_argument("--unfreeze", help="not freezed layers", default=-1)
    parser.add_argument("--id_exp", help="seed", default=1)
    parser.add_argument("--checkpointer", "-c", help="seed", action='store_true')
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
    fold_list = list(range(cv)) if isinstance(cv, int) else [int(value) for value in os.listdir(cfg['data']['fold_dir'])]
    print(fold_list)
    # Data config
    data_cfg = cfg['data']['modes']['img']
    CV = '_' + str(cfg['data']['cv'])
    CLAHE = '_Clahe' if data_cfg['preprocess']['clahe'] else ''
    Filter = '_Filter3th' if data_cfg['preprocess']['filter'] else ''
    Clip = '_Clip2-98' if data_cfg['preprocess']['clip'] else ''
    Drop = f'_Drop{cfg["model"]["dropout_rate"]}'

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        Batch = f'_Batch{cfg["data"]["batch_size"] // torch.cuda.device_count()}'
    else:
        Batch = f'_Batch{cfg["data"]["batch_size"]}'
    LearningRate = f'_LR{cfg["trainer"]["optimizer"]["lr"]}'
    Masked = '_LungMask' if data_cfg['preprocess']['masked'] else '_Entire'
    bbox_resize = '_LungBbox' if data_cfg['preprocess']['bbox_resize'] else '_Entire' if cfg['data']['modes']['img']['bbox_resize'] else ''
    softmax = '_Softmax' if cfg['model']['softmax'] else ''
    freezing = '_unfreeze_' if not cfg['model']['freezing'] else ''
    warming = f'_warmup_' if cfg['trainer']['warmup_epochs'] != 0 else ''
    loss = f'_loss_{cfg["trainer"]["loss"]}' if cfg['trainer']['loss'].lower() != 'mse' else ''

    # Experiment name
    exp_name = cfg['exp_name'] + CV + Batch + LearningRate + warming + loss + Drop + softmax + CLAHE + Filter + Clip + Masked + bbox_resize + freezing
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
    assert model_name in [
        "vgg11", "shufflenet_v2_x1_5", "squeezenet1_0", "squeezenet1_1", "mobilenet_v2",
        "vgg11_bn", "densenet121_CXR",
        "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "densenet121", "densenet121_CXR", "densenet169", "densenet161", "densenet201", "googlenet", "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0", "resnext50_32x4d", "wide_resnet50_2", "mnasnet0_5", "mnasnet1_0"]
    model_dir = os.path.join(cfg['data']['model_dir'], exp_name, model_name)  # folder to save model
    print(' ----------| Model directory: ', model_dir)
    if not args.checkpointer and os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    mkdir(model_dir)
    report_dir = os.path.join(cfg['data']['report_dir'], exp_name, model_name)  # folder to save results
    print(' ----------| Report directory: ', report_dir)
    if not args.checkpointer and os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    mkdir(report_dir)


    logger = Logger(file_name=os.path.join(report_dir, f'log_print_out_{model_name}.txt'), file_mode="w", should_flush=True)
    with open(os.path.join(report_dir, f'config_{model_name}.yaml'), 'w') as file:
        documents = yaml.dump(cfg, file)
    plot_training_dir = os.path.join(report_dir, "training_plot")

    if not args.checkpointer and os.path.exists(plot_training_dir):
        shutil.rmtree(plot_training_dir)
    mkdir(plot_training_dir)

    # REPORT FINAL:
    # 1) VALIDATION
    final_results_validation = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])

    # 2) TEST
    final_results_test = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'])
    # ------------------- FOLD ITERATION -------------------
    for fold in fold_list:
        string_fold = '-----------| Fold ' + str(fold) + ' |----------'
        print(''.center(len(string_fold), '-'))
        print(''.center(len(string_fold), '-'))
        print(string_fold)
        print(''.center(len(string_fold), '-'))
        print(''.center(len(string_fold), '-'))

        # Directories model
        model_fold_dir = os.path.join(model_dir, str(fold))
        mkdir(model_fold_dir)
        plot_training_fold_dir = os.path.join(plot_training_dir, str(fold))
        mkdir(plot_training_fold_dir)
        # Directories reports inference
        val_results_by_patient = os.path.join(report_dir, 'val_prediction', str(fold))
        if os.path.exists(val_results_by_patient):
            shutil.rmtree(val_results_by_patient)
        mkdir(val_results_by_patient)
        test_results_by_patient = os.path.join(report_dir, 'test_prediction', str(fold))
        if os.path.exists(test_results_by_patient):
            shutil.rmtree(test_results_by_patient)
        mkdir(test_results_by_patient)

        # Data Loaders for MORBIDITY TASK
        fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], str(fold), '%s.txt' % step), delimiter=" ") for step in steps}

        if is_debug():
            pass
            fold_data['train'] = fold_data['train']
            fold_data['val'] = fold_data['val'][100:280]
            fold_data['test'] = fold_data['test'][100:280]
        # ------------------- DATA -------------------
        datasets = {step: DatasetImgAFC(data=fold_data[step], classes=classes, cfg=cfg['data']['modes']['img'], step=step) for step in steps}

        # Data loaders
        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)}
        # Idx to class name
        idx_to_class = {v: k for k, v in datasets['train'].class_to_idx.items()}
        # ------------------- MODEL -------------------
        model = get_SingleTaskModel(backbone=model_name, cfg=cfg, device=device)
        # allocate model on gpu
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model, [0, 1])

        # Load pretrained model if exists
        if cfg['model']['pretrained']:
            model.load_state_dict(torch.load(os.path.join(cfg['model']['pretrained'], str(fold), "model.pt"), map_location=device))

        # Checkpointer
        model_trained = False
        if args.checkpointer:
            list_saved_model = glob.glob(os.path.join(model_fold_dir, "*.pt"))
            if list_saved_model.__len__() > 0:
                print("Loading model from checkpoint for the fold ", fold)  # load the last checkpoint with the best model
                model_loaded = torch.load(list_saved_model[-1], map_location=device)

                if isinstance(model_loaded, torch.nn.DataParallel):
                    model_dict = model_loaded.module.state_dict()
                elif isinstance(model_loaded, torch.nn.Module):
                    model_dict = model_loaded.state_dict()
                if isinstance(model, torch.nn.DataParallel):
                    model.module.load_state_dict(model_dict)
                elif isinstance(model, torch.nn.Module):
                    model.load_state_dict(model_dict)
                model_trained = True

        model = model.to(device)

        # Loss function
        if cfg['trainer']['loss'].lower() == 'mse':
            criterion = nn.MSELoss().to(device)
        elif cfg['trainer']['loss'].lower() == 'bce' and not softmax:
            criterion = nn.BCEWithLogitsLoss().to(device)
        elif cfg['trainer']['loss'].lower() == 'bce':
            criterion = nn.BCELoss().to(device)

        if not model_trained:
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['optimizer']['lr'], weight_decay=cfg['trainer']['optimizer']['weight_decay'])
            # LR Scheduler

            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode=cfg['trainer']['scheduler']['mode'],
                                                       patience=cfg['trainer']['scheduler']['patience'],
                                                       min_lr=float(cfg['trainer']['scheduler']['min_lr']),
                                                       factor=cfg['trainer']['scheduler']['factor'])

            # Train model

            model, history = train_morbidity(model=model,
                                             cfg=cfg,
                                             criterion=criterion,
                                             model_file_name=f'model_{model_name}',
                                             dataloaders=data_loaders,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             num_epochs=cfg['trainer']['max_epochs'],
                                             max_epochs_stop=cfg['trainer']['early_stopping'],
                                             model_dir=model_fold_dir,
                                             device=device)

            # Plot Training
            plot_training(history=history, plot_training_dir=plot_training_fold_dir)

        # ------------------- VALIDATION -------------------
        # Evaluate the model on all the validation data
        results_validation_by_image, results_classes_val, common_metrics_val = (
            evaluate_morbidity(model=model,
                               test_loader=data_loaders['val'],
                               criterion=criterion,
                               idx_to_class=idx_to_class,
                               device=device,
                               cfg=cfg,
                               topk=(1,)))

        final_results_validation = (
            compute_report_metrics(final_report_folds=final_results_validation,
                                   metrics_report=common_metrics_val,
                                   classes_report=results_classes_val,
                                   classes=classes,
                                   fold=fold,
                                   results_by_patient=results_validation_by_image,
                                   model_name=model_name,
                                   report_path=val_results_by_patient))

        # ------------------- TEST -------------------
        # Evaluate the model on all the test data
        results_test_by_image, results_classes_test, common_metrics_test = evaluate_morbidity(
            model=model,
            test_loader=data_loaders['test'],
            criterion=criterion,
            idx_to_class=idx_to_class,
            device=device,
            cfg=cfg,
            topk=(1,))
        final_results_test = (
            compute_report_metrics(final_report_folds=final_results_test,
                                   metrics_report=common_metrics_test,
                                   classes_report=results_classes_test,
                                   classes=classes,
                                   fold=fold,
                                   results_by_patient=results_test_by_image,
                                   model_name=model_name,
                                   report_path=test_results_by_patient))

        # Test model
        print(final_results_test)

    # SAVE FINAL RESULTS FOR ALL THE METRICS
    # TEST
    final_results_test.to_excel(os.path.join(report_dir, f'[all]_test_results_{model_name}.xlsx'))


if __name__ == '__main__':
    main()
