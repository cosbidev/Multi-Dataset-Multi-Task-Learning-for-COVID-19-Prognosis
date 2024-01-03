import glob
import sys
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

print('Python %s on %s' % (sys.version, sys.platform))
print(os.getcwd(), ' the current working directory')
sys.path.extend('./')
import torch
import torch.nn as nn
import pandas as pd
import yaml
from src import mkdir, seed_all, DatasetImgAFC, seed_worker, get_SingleTaskModel


# Configuration file


def main():
    # ID experiment
    id_exp = 'BINA-r2'

    """
    'configs/5/morbidity/afc_config_singletask_cv5.yaml',
                         'configs/5/morbidity/afc_config_singletask_Masked_cv5.yaml',
    """

    for cfg_file in [
                     'configs/5/morbidity/afc_config_singletask_cv5.yaml',
                     'configs/loCo/morbidity/afc_config_singletask_loCo.yaml',
                     ]:
        # Load configuration file
        with open(cfg_file) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        with open('configs/common/common_morbidity.yaml') as file_common:
            cfg_common = yaml.load(file_common, Loader=yaml.FullLoader)
        cfg.update(cfg_common)
        del cfg_common
        # Seed everything
        seed_all(cfg['seed'])

        # Parameters
        batch_size = cfg['trainer']['batch_size']
        classes = cfg['data']['classes']

        model_list = ['efficientnet_es_pruned',
 'resnet18',
 'mobilenet_v2',
 'resnet50_ImageNet_ChexPert',
 'efficientnet_es',
 'resnet50_ImageNet_ChestX-ray14',
 'resnet50',
 'resnet50_ChexPert',
 'densenet121',
 'resnext50_32x4d',
 'resnet50_ChestX-ray14',
 'densenet121_CXR',
 'shufflenet_v2_x0_5',
 'wide_resnet50_2',
 'googlenet',
 'shufflenet_v2_x1_0',
 'efficientnet_lite0',
 'resnet34',
 'resnet101',
 'efficientnet_b0',
 'efficientnet_b1_pruned',]

        steps = ['train']
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
            Batch = f'_Batch{batch_size // torch.cuda.device_count()}'
        else:
            Batch = f'_Batch{batch_size}'
        LearningRate = f'_LR{cfg["trainer"]["optimizer"]["lr"]}'
        Masked = '_LungMask' if data_cfg['preprocess']['masked'] else '_Entire'
        bbox_resize = '_LungBbox' if data_cfg['preprocess']['bbox_resize'] else '_Entire' if cfg['data']['modes']['img']['bbox_resize'] else ''
        softmax = '_Softmax' if cfg['model']['softmax'] else ''
        freezing = '_freezeBB_' if cfg['model']['freezing'] else ''
        warming = f'_warmup_' if cfg['trainer']['warmup_epochs'] != 0 else ''
        loss = f'_loss_{cfg["trainer"]["loss"]}' if cfg['trainer']['loss'].lower() != 'mse' else ''
        # Experiment name
        exp_name = cfg['exp_name'] + CV + Batch + LearningRate + warming + loss + Drop + softmax + CLAHE + Filter + Clip + freezing + Masked + bbox_resize
        print(' ----------| Experiment name: ', exp_name)
        # Device
        device = torch.device(cfg['device']['cuda_device'] if torch.cuda.is_available() else "cpu")
        num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
        print(device)
        # Directories
        cfg['exp_name'] = cfg['exp_name'] + f'_{id_exp}'
        cfg['data']['model_dir'] = os.path.join(cfg['data']['model_dir'], cfg['exp_name'])  # folder to save trained model
        cfg['data']['report_dir'] = os.path.join(cfg['data']['report_dir'], cfg['exp_name'])

        # folder to save model
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

            # Directories reports inference
            # Data Loaders for MORBIDITY TASK

            fold_data = {step: pd.read_csv(os.path.join(cfg['data']['fold_dir'], str(fold), '%s.txt' % step), delimiter=" ") for step in steps}
            datasets = {step: DatasetImgAFC(data=fold_data[step], classes=classes, cfg=cfg, step=step) for step in steps}
            # ------------------- MODEL -------------------

            for model_name in model_list:

                # Files and Directories
                model_dir = os.path.join(cfg['root'],cfg['data']['model_dir'], exp_name, model_name)
                report_dir = os.path.join(cfg['data']['report_dir'], exp_name, model_name)
                model_fold_dir = os.path.join(model_dir, str(fold))
                print('-------------------')
                print('MODEL-DIR: ', model_dir)
                print('REPORT-DIR: ', report_dir)
                print('MODEL-FOLD-DIR: ', model_fold_dir)
                print('-------------------')


                model = get_SingleTaskModel(backbone=model_name, cfg=cfg, device=device)
                # Data loaders
                data_loaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker)}

                # Load model form directory
                if True:
                    list_saved_model = glob.glob(os.path.join(model_fold_dir, "*.pt"))
                    if list_saved_model.__len__() > 0:
                        print("Loading model from checkpoint for the fold ", fold)  # load the last checkpoint with the best model
                        model_loaded = torch.load(list_saved_model[-1], map_location=device)

                        if isinstance(model_loaded, torch.nn.DataParallel):
                            model_dict = model_loaded.module.state_dict()
                        elif isinstance(model_loaded, torch.nn.Module):
                            model_dict = model_loaded.state_dict()
                        elif isinstance(model_loaded, dict):
                            model_dict = model_loaded
                        if isinstance(model, torch.nn.DataParallel):
                            model.module.load_state_dict(model_dict)
                        elif isinstance(model, torch.nn.Module):
                            model.load_state_dict(model_dict)

                    else:
                        print("No checkpoint found for the fold ", fold)
                        continue

                model = model.to(device)

                # Loss function
                if cfg['trainer']['loss'].lower() == 'mse':
                    criterion = nn.MSELoss().to(device)
                elif cfg['trainer']['loss'].lower() == 'bce':
                    criterion = nn.BCELoss().to(device)

                # ------------------- TRAIN PREDICTION -------------------
                # Evaluate the model on all the validation data
                true_labels = []
                predicted_labels = []
                model.eval()



                with torch.no_grad():
                    results_for_images = pd.DataFrame(columns=['Predicted', 'labels', 'probs', 'Loss', 'Correct', 'pz_name'])
                    # Testing loop
                    for outs in tqdm(data_loaders['train'], total=len(data_loaders['train'])):
                        data = outs[0]
                        targets = outs[1]
                        file_name = outs[2]
                        data = data.to(device)
                        targets = targets.to(device)
                        # Raw model output
                        outputs = model(data.float())
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                            targets = targets[:, :4]

                        values_probs, preds = torch.max(outputs, 1)
                        _, labels_gt = torch.max(targets, 1)

                        true_labels.append(labels_gt.data.cpu().numpy())
                        predicted_labels.append(preds.data.cpu().numpy())

                        # Iterate through each example
                        new_info = pd.DataFrame(columns= ['Predicted', 'labels', 'probs', 'Loss', 'Correct', 'pz_name'])


                        for pred, true, target, file_name_pz, prob, output  in zip(preds, labels_gt, targets, file_name, values_probs, outputs):
                            # Calculate the loss
                            loss = criterion(output, target)

                            # Append loss:
                            new_info.loc[file_name_pz, 'Loss'] = loss.item()
                            # group by patient and by checkiung if theit prediction is good

                            new_info.loc[file_name_pz, 'probs'] = prob.item()
                            new_info.loc[file_name_pz, 'Predicted'] = pred.item()
                            new_info.loc[file_name_pz, 'labels'] = true.item()
                            new_info.loc[file_name_pz, 'pz_name'] = file_name_pz
                            new_info.loc[file_name_pz, 'Correct'] = 1 if pred.equal(true) else 0


                            results_for_images.loc[len(results_for_images)] = new_info.loc[file_name_pz]


                # DATA to separate for the Quantile Sampling by class
                data_to_split = results_for_images.set_index('pz_name', drop=False)
                data_to_split = data_to_split.sort_index()

                # ------------------- QUANTILE SAMPLING -------------------
                list_q_class = []
                for label_correct in results_for_images.loc[:, 'Correct'].unique():

                    # Class label 0 and label for correct and incorrect images
                    data_to_plot_class_0 = data_to_split[(data_to_split['Correct'] == label_correct) & (data_to_split['labels'] == 0)]

                    # Quantile Sampling Variable: Probability
                    quantile_class_0 = data_to_plot_class_0[['probs']]

                    # Pandas Quantile function to split the data in 3 quantiles
                    q_0 = pd.qcut(quantile_class_0.values[:, 0].astype(float), q=3, labels=False, duplicates='drop')

                    # Assign the quantile to the dataframe
                    data_to_plot_class_0 = data_to_plot_class_0.assign(q=q_0)


                    # Class label Severe (1) and label for correct and incorrect classified images
                    data_to_plot_class_1 = data_to_split[(data_to_split['Correct'] == label_correct) & (data_to_split['labels'] == 1)]

                    # Quantile Sampling Variable: Probability
                    quantile_class_1 = data_to_plot_class_1[['probs']]

                    # Pandas Quantile function to split the data in 3 quantiles
                    q_1 = pd.qcut(quantile_class_1.values[:, 0].astype(float), q=3, labels=False, duplicates='drop')

                    # Assign the quantile to the dataframe
                    data_to_plot_class_1 = data_to_plot_class_1.assign(q=q_1)


                    # Concatenate the two dataframes
                    data_class_0_q = data_to_plot_class_0[['q', 'pz_name']].set_index('pz_name', drop=False)
                    data_class_1_q = data_to_plot_class_1[['q', 'pz_name']].set_index('pz_name', drop=False)
                    data_class_all_q = pd.concat([data_class_0_q, data_class_1_q], axis=0)


                    data_class_all_q = data_class_all_q.sort_index()
                    # Append to the list
                    list_q_class.append(data_class_all_q)


                # Concatenate the list of dataframes
                list_q_class_all_prediction = pd.concat(list_q_class, axis=0)
                list_q_class_all_prediction = list_q_class_all_prediction.sort_index()

                # Add the list_q_class_all_prediction to the results_for_images as a column

                results_for_images_q = pd.concat([data_to_split, list_q_class_all_prediction], axis=1).drop(columns=['pz_name'])
                results_for_images_q['pz_name'] = results_for_images_q.index

                # ------------------- SAVE -------------------
                q_dir = os.path.join(report_dir, 'quantile_sampling', str(fold))
                mkdir(q_dir)
                results_for_images_q.to_csv(os.path.join(q_dir, 'q_train_images.csv'))


                # ------------------- PLOT -------------------


                # Scatter plot of the data points, with color coding for each cluster
                # plt.scatter(data_to_cluster['probs'], data_to_cluster['Loss'], c=data_to_cluster['Cluster'], cmap='viridis', marker='o')
                colors = ['blue', 'yellow']
    
                """  
                f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 5))
    
                f.set_figheight(15)
                f.set_figwidth(15)
    
    
                # Markers Parameters
                marker_dict = {0: 'o', 1: '^', 2: '.'}
                correct_scale = {0: 'olive', 1: 'yellowgreen', 2: 'lawngreen'}
                wrong_scale = {0: 'silver', 1: 'gray', 2: 'black'}
    
                for label_correct in results_for_images.loc[:, 'Correct'].unique():
    
                    # Class label 0 and label for correct and incorrect images
                    data_to_plot_class_0 = results_for_images_q[(results_for_images_q['Correct'] == label_correct) & (results_for_images_q['labels'] == 0)]
    
                    # Quantile Sampling Variable: Probability
                    q_0 = data_to_plot_class_0['q']
    
    
                    for q_0_i in np.unique(q_0):
                        ax1.scatter(data_to_plot_class_0[data_to_plot_class_0['q'] == q_0_i]['probs'], data_to_plot_class_0[data_to_plot_class_0['q'] == q_0_i]['Loss'],
                                    color=correct_scale[q_0_i] if label_correct == 1 else wrong_scale[q_0_i], label='Correct' if label_correct == 1 else 'Wrong', marker=marker_dict[q_0_i],
                                    alpha=0.5, s=100
                                    )
    
    
                    ax1.set_title('Class Mild Correct vs Wrong predictions')
                    ax1.legend()
                    ax1.grid(True)
    
                    # Class label 1 and label for correct and incorrect images
                    data_to_plot_class_1 = results_for_images_q[(results_for_images_q['Correct'] == label_correct) & (results_for_images_q['labels'] == 1)]
    
                    # Quantile Sampling Variable: Probability
                    q_1 = data_to_plot_class_1['q']
    
                    for q_1_i in np.unique(q_1):
                        ax2.scatter(data_to_plot_class_1[data_to_plot_class_1['q'] == q_1_i]['probs'], data_to_plot_class_1[data_to_plot_class_1['q'] == q_1_i]['Loss'],
                                    color=correct_scale[q_1_i] if label_correct == 1 else wrong_scale[q_1_i], label='Correct' if label_correct == 1 else 'Wrong',
                                    marker=marker_dict[q_1_i], alpha=0.5, s=100)
                    ax2.set_title('Class Severe Correct vs Wrong predictions')
                    ax2.legend()
                    ax2.grid(True)
    
                plt.suptitle('By Classes Plot Probs vs Loss')
                plt.xlabel('Probability')
                plt.ylabel('Loss')
    
                # Show the centroids
                plt.legend()
    
                pass"""

if __name__ == '__main__':
    main()
