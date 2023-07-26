import argparse
import sys
import os

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
from src import mkdir, seed_all, DatasetImgAFC, seed_worker, get_SingleTaskModel, plot_training, \
    train_single, evaluate
from src.utils.utils_aiforcovid import *

# Configuration file



Models = [
"alexnet",
"resnet18",
"resnet34",
"resnet50",
"resnet101",
"resnet152",
"densenet121",
"densenet169",
"densenet161",
"densenet201",
"shufflenet_v2_x0_5",
"shufflenet_v2_x1_0",
"shufflenet_v2_x1_5",
"mobilenet_v2",
"resnext50_32x4d",
"wide_resnet50_2",
"mnasnet0_5",
"mnasnet1_0",
"vgg11",
"vgg11_bn",
"vgg13",
"vgg13_bn",
"vgg16",
"vgg16_bn",
"vgg19",
"vgg19_bn"
]




def main():
    # Configuration file
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("--cfg_file", help="Number of folder", type=str)
    parser.add_argument("--model_name", help="model_name", choices=Models)
    parser.add_argument("--output_dir", help="output directory path", default="data/processed", required=True)
    parser.add_argument("--input_data", help="input directory path for data", default="data/processed", required=True)
    parser.add_argument("--unfreeze", help="not freezed layers", default=-1)
    parser.add_argument("--id_exp", help="seed", default=1)
    args = parser.parse_args()

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
    Batch = f'_Batch{cfg["data"]["batch_size"]}'
    LearningRate = f'_LR{cfg["trainer"]["optimizer"]["lr"]}'
    Masked = '_LungMask' if data_cfg['preprocess']['masked'] else '_Entire'
    bbox_resize = '_LungBbox' if data_cfg['preprocess']['bbox_resize'] else '_Entire' if cfg['data']['modes']['img']['bbox_resize'] else ''
    # Experiment name
    exp_name = cfg['exp_name'] + CV + Batch + LearningRate + Drop + CLAHE + Filter + Clip + Masked + bbox_resize



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
              "vgg11",
              "vgg11_bn",
              "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
              "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "squeezenet1_0", "squeezenet1_1",
              "densenet121", "densenet169", "densenet161", "densenet201", "googlenet", "shufflenet_v2_x0_5",
              "shufflenet_v2_x1_0", "mobilenet_v2", "resnext50_32x4d", "wide_resnet50_2", "mnasnet0_5", "mnasnet1_0"]
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
    report_file_temp = os.path.join(report_dir, 'report_' + str(cv) + '_temp.xlsx')



    # Results table
    results_frame = {}
    acc_cols = []
    acc_cat_cols = collections.defaultdict(lambda: [])
    for fold in fold_list:
        acc_col = str(fold) + " ACC"
        acc_cols.append(acc_col)
        results_frame[acc_col] = []
        for cat in classes:
            cat_col = str(fold) + " ACC " + cat
            acc_cat_cols[cat].append(cat_col)
            results_frame[cat_col] = []
    acc_cat_cols = dict(acc_cat_cols)



    for fold in fold_list:
        # Dir
        model_fold_dir = os.path.join(model_dir, str(fold))
        mkdir(model_fold_dir)
        plot_training_fold_dir = os.path.join(plot_training_dir, str(fold))
        mkdir(plot_training_fold_dir)
        plot_test_fold_dir = os.path.join(plot_test_dir, str(fold))
        mkdir(plot_test_fold_dir)


        # Data Loaders for MORBIDITY TASK
        fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], str(fold), '%s.txt' % step), delimiter=" ") for step in steps}
        datasets = {step: DatasetImgAFC(data=fold_data[step], classes=classes, cfg=cfg['data']['modes']['img'], step=step) for step in steps}

        data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker),
                        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker),
                        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)}
        #
        idx_to_class = {v: k for k, v in datasets['train'].class_to_idx.items()}
        # Model
        #input, _, _ = next(iter(data_loaders["train"]))
        model = get_SingleTaskModel(backbone=model_name, cfg=cfg, device=device)
        if cfg['model']['pretrained']:
            model.load_state_dict(torch.load(os.path.join(cfg['model']['pretrained'], str(fold), "model.pt"), map_location=device))
        model = model.to(device)

        # Loss function
        criterion = nn.MSELoss().to(device)
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg['trainer']['optimizer']['lr'], weight_decay=cfg['trainer']['optimizer']['weight_decay'])
        # LR Scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['trainer']['scheduler']['mode'], patience=cfg['trainer']['scheduler']['patience'])
        # Train model

        model, history = train_single(model=model,
                                      criterion=criterion,
                                      model_file_name=f'model_{model_name}.pt',
                                      dataloaders=data_loaders,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      num_epochs=cfg['trainer']['max_epochs'],
                                      max_epochs_stop=cfg['trainer']['early_stopping'],
                                      model_dir=model_fold_dir,
                                      device=device)


        # Plot Training
        plot_training(history=history, plot_training_dir=plot_training_fold_dir)
        # Evaluate the model on all the test data
        results, acc = evaluate(model, data_loaders['test'], criterion, idx_to_class, device, topk=(1, ))
        # Test model




        print(results)
        print(acc)

        # Update report
        results_frame[str(fold) + " ACC"].append(acc)
        for cat in classes:
            results_frame[str(fold) + " ACC " + str(cat)].append(results.loc[results["class"] == cat]["top1"].item())

        # Save temporary Results
        results_frame_temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_frame.items()]))
        for cat in classes[::-1]:
            results_frame_temp.insert(loc=0, column='std ACC ' + cat,
                                      value=results_frame_temp[acc_cat_cols[cat]].std(axis=1))
            results_frame_temp.insert(loc=0, column='mean ACC ' + cat,
                                      value=results_frame_temp[acc_cat_cols[cat]].mean(axis=1))
        results_frame_temp.insert(loc=0, column='std ACC', value=results_frame_temp[acc_cols].std(axis=1))
        results_frame_temp.insert(loc=0, column='mean ACC', value=results_frame_temp[acc_cols].mean(axis=1))
        results_frame_temp.insert(loc=0, column='model', value=model_name)
        results_frame_temp.to_excel(report_file_temp, index=False)

    results_frame = pd.DataFrame.from_dict(results_frame)
    for cat in classes[::-1]:
        results_frame.insert(loc=0, column='std ACC ' + cat, value=results_frame[acc_cat_cols[cat]].std(axis=1))
        results_frame.insert(loc=0, column='mean ACC ' + cat, value=results_frame[acc_cat_cols[cat]].mean(axis=1))
    results_frame.insert(loc=0, column='std ACC', value=results_frame[acc_cols].std(axis=1))
    results_frame.insert(loc=0, column='mean ACC', value=results_frame[acc_cols].mean(axis=1))
    results_frame.insert(loc=0, column='model', value=model_name)
    results_frame.to_excel(report_file, index=False)

if __name__ == '__main__':
    main()