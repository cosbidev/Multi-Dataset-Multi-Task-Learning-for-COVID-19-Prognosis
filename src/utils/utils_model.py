import itertools
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import pandas as pd
from scipy.stats import pearsonr
from skimage.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, r2_score
from tqdm import tqdm
import os
from skimage import color
import math
from torchvision import models
from .utils_general import runcmd, mkdir

model_list = ["alexnet",
              "vgg11",
              "vgg11_bn",
              "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
              "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "squeezenet1_0", "squeezenet1_1",
              "densenet121", "densenet169", "densenet161", "densenet201", "googlenet", "shufflenet_v2_x0_5",
              "shufflenet_v2_x1_0", "mobilenet_v2", "resnext50_32x4d", "wide_resnet50_2", "mnasnet0_5", "mnasnet1_0"]


def freeze_backbone(model, perc=0.5):
    start_counter_seq = 0
    # Freeze Backbone
    Portion_to_freeze = 1 / perc
    layers_seq_to_freeze = np.ceil(
        len([module for module in list(model.backbone.modules())[1:] if isinstance(module, nn.Sequential)]) / Portion_to_freeze)
    for module in list(model.backbone.modules())[1:]:
        print('ANALIZE THIS: ', module.__class__)
        if isinstance(module, nn.Sequential):
            start_counter_seq += 1
            for i, param in enumerate(list(module.parameters())):
                param.requires_grad = False
        for i, param in enumerate(list(module.parameters())):
            param.requires_grad = False
        if start_counter_seq == layers_seq_to_freeze:
            break
    return model


def change_head(model, new_head, model_name):
    """
    This function changes the head for a selected model.
    Args:
        model: model by torch hub
        new_head: Sequential  model by torch
        model_name: model string

    Returns: the model with a modified head

    """
    if "resnet" in model_name or "shufflenet" in model_name or "resnext" in model_name or "googlenet" in model_name:
        model.fc = new_head
    elif "vgg" in model_name or "mobilenet" in model_name or "alexnet" in model_name:
        model.classifier[-1] = new_head
    elif "densenet" in model_name:
        model.classifier = new_head
    elif "squeezenet" in model_name:
        model.classifier[1] = new_head
    elif "efficientnet" in model_name:
        model.classifier = new_head
    elif "vit" in model_name:
        model.head = new_head


# Try To TODO these experiments
vit_eff_BACKBONES = ['efficientnet_lite0',
                     'efficientnet_b0',
                     'efficientnet_b0_gn',
                     'efficientnet_lite1',
                     'efficientnet_es',
                     'vit_medium_patch16_gap_256',
                     'vit_medium_patch16_reg4_gap_256',
                     'vit_medium_patch16_reg4_256']


def get_backbone(model_name=''):
    # Finetuning the convnet
    print("********************************************")
    model, in_features = None, 0

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features





    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "resnet50_ChestX-ray14":
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
    elif model_name == "resnet50_ChexPert":
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
    elif model_name == "resnet50_ImageNet_ChestX-ray14":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "resnet50_ImageNet_ChexPert":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
    elif model_name == "densenet121_CXR":
        model = models.densenet121(pretrained=False)
        in_features = model.classifier.in_features
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=True)
        in_features = model.classifier.in_features
        # model.classifier = nn.Linear(in_features=1664, out_features=len(class_names), bias=True)
    elif model_name == "densenet161":
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features
        # model.classifier = nn.Linear(in_features=2208, out_features=len(class_names), bias=True)
    elif model_name == "densenet201":
        model = models.densenet201(pretrained=True)
        in_features = model.classifier.in_features
        # model.classifier = nn.Linear(in_features=1920, out_features=len(class_names), bias=True)
    elif model_name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(pretrained=True)
        in_features = model.fc.in_features
        # model.fc = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        in_features = model.fc.in_features
        # model.fc = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
    elif model_name == "shufflenet_v2_x1_5":
        model = models.shufflenet_v2_x1_5(pretrained=True)
        in_features = model.fc.in_features
        # model.fc = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        in_features = 1280
        # model.classifier[-1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=True)
        in_features = model.fc.in_features
        # model.fc = nn.Linear(2048, len(class_names), bias=True)
    elif model_name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
        in_features = model.classifier[1].in_channels
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
        in_features = model.classifier[1].in_channels
    elif model_name == "vgg11":
        model = models.vgg11(pretrained=True)
        in_features = model.classifier[-1].in_features
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
        in_features = model.classifier[1].in_features
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
        in_features = model.fc.in_features
        # model.fc = nn.Linear(2048, len(class_names), bias=True)
    elif model_name == "vgg11_bn":
        model = models.vgg11_bn(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
    elif model_name == "vgg13":
        model = models.vgg13(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
    elif model_name == "vgg13_bn":
        model = models.vgg13_bn(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
    elif model_name == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)

    elif "efficientnet" in model_name:
        model = timm.create_model(model_name, pretrained=True)
        in_features = model.classifier.in_features
    elif "vit" in model_name:
        model = timm.create_model(model_name, pretrained=False)
        in_features = model.head.in_features
    else:
        raise ValueError("Invalid model name")

    assert model is not None
    return model, in_features


def update_learning_rate(optimizer, scheduler, metric=None):
    """Update learning rates for all the networks; called at the end of every epoch"""
    old_lr = optimizer.param_groups[0]['lr']
    print('Metric Value', metric)
    scheduler.step(float(metric))
    lr = optimizer.param_groups[0]['lr']
    print('optimizer: %.7s  --learning rate %.7f -> %.7f' % (
        optimizer.__class__.__name__, old_lr, lr) if not old_lr == lr else 'Learning rate non modificato: %s' % (old_lr))


def get_metrics_classification_severity(y_true, y_pred, regression_type='area'):
    if regression_type == 'area':
        y_sampling_pred = np.zeros((y_pred.shape[0], y_pred.shape[1] + 3))
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                if y_pred[i, j] <= 0.5:
                    y_sampling_pred[i, j] = 0
                elif 0.5 < y_pred[i, j] <= 1.5:
                    y_sampling_pred[i, j] = 1
                elif 1.5 < y_pred[i, j] <= 2.5:
                    y_sampling_pred[i, j] = 2
                elif 2.5 < y_pred[i, j]:
                    y_sampling_pred[i, j] = 3

        for row in range(y_sampling_pred.shape[0]):
            y_sampling_pred[row, 6] = np.sum(y_sampling_pred[row, :3])
            y_sampling_pred[row, 7] = np.sum(y_sampling_pred[row, 3:6])
            y_sampling_pred[row, 8] = np.sum(y_sampling_pred[row, :6])

        max_value_l1 = np.sum(y_true[:, -1])
        max_value_exp = np.sum([np.sum([np.exp(y_true[row, i]) for i in range(6)]) for row in range(y_true.shape[0])])
        max_value_s = np.sum([np.sum([y_true[row, i] ** 2 for i in range(6)]) for row in range(y_true.shape[0])])
        min_value = 0

        error_global = np.sum([np.abs(y_true[row, 8] - y_sampling_pred[row, 8]) for row in range(y_true.shape[0])])

        errors_l1 = [np.sum([np.abs(y_true[row, column] - y_sampling_pred[row, column]) for column in range(6)]) for row in range(y_true.shape[0])]
        errors_exp = [np.sum([np.exp(np.abs(y_true[row, column] - y_sampling_pred[row, column])) for column in range(6)]) for row in range(y_true.shape[0])]
        errors_squared = [np.sum([(y_true[row, column] - y_sampling_pred[row, column]) ** 2 for column in range(6)]) for row in range(y_true.shape[0])]

        # ERRORS COMPUTATION:
        # distance L1 between predictions and true values for each area
        Normalized_error_l1 = (np.sum(errors_l1) - min_value) / (max_value_l1 - min_value)
        # distance L2 between predictions and true values for each area
        Normalized_error_squared = (np.sum(errors_squared) - min_value) / (max_value_s - min_value)
        # distance exp between predictions and true values for each area
        Normalized_error_exp = (np.sum(errors_exp) - min_value) / (max_value_exp - min_value)
        # ACCURACY COMPUTATION:
        # accuracy distance exp between predictions and true values for each area
        accuracy_distance_exp = 1 - Normalized_error_exp
        # accuracy distance L1 between predictions and true values for each area
        accuracy_distance_l1 = 1 - Normalized_error_l1
        # accuracy distance L2 between predictions and true values for each area
        accuracy_distance_squared = 1 - Normalized_error_squared
        # Accuracy boolean for Global Score
        accuracy_global_accuracy = sum(list(y_sampling_pred[:, -1] == y_true[:, -1])) / y_true.shape[0]
        # Accuracy boolean for Lung R and Lung L
        acc_lung_l = sum(list(y_sampling_pred[:, 7] == y_true[:, 7])) / y_true.shape[0]
        acc_lung_r = sum(list(y_sampling_pred[:, 6] == y_true[:, 6])) / y_true.shape[0]
        # Accuracy boolean for each area
        acc_area_1 = sum(list(y_sampling_pred[:, 0] == y_true[:, 0])) / y_true.shape[0]
        acc_area_2 = sum(list(y_sampling_pred[:, 1] == y_true[:, 1])) / y_true.shape[0]
        acc_area_3 = sum(list(y_sampling_pred[:, 2] == y_true[:, 2])) / y_true.shape[0]
        acc_area_4 = sum(list(y_sampling_pred[:, 3] == y_true[:, 3])) / y_true.shape[0]
        acc_area_5 = sum(list(y_sampling_pred[:, 4] == y_true[:, 4])) / y_true.shape[0]
        acc_area_6 = sum(list(y_sampling_pred[:, 5] == y_true[:, 5])) / y_true.shape[0]
        # Accuracy boolean for score
        acc_all_ = {'acc_boolean_accuracy': accuracy_global_accuracy,
                    'acc_lung_r': acc_lung_r,
                    'acc_lung_l': acc_lung_l,
                    'acc_a': acc_area_1,
                    'acc_b': acc_area_2,
                    'acc_c': acc_area_3,
                    'acc_d': acc_area_4,
                    'acc_e': acc_area_5,
                    'acc_f': acc_area_6
                    }
        return (accuracy_distance_l1,
                accuracy_distance_exp,
                accuracy_distance_squared,
                y_sampling_pred,
                acc_all_)
    elif regression_type == 'consistent':
        score_pred_RR = np.sum(y_pred[:, :3], axis=1)
        score_pred_LL = np.sum(y_pred[:, 3:6], axis=1)
        score_pred_G = np.sum(y_pred, axis=1)

        accuracy_RR = 100 * (np.sum(y_true[:, 6] == score_pred_RR) / y_true.shape[0])
        accuracy_LL = 100 * (np.sum(y_true[:, 7] == score_pred_LL) / y_true.shape[0])
        accuracy_G = 100 * (np.sum(y_true[:, 8] == score_pred_G) / y_true.shape[0])

        max_value_l1 = np.sum(y_true[:, -1])
        max_value_exp = np.sum([np.sum([np.exp(y_true[row, i]) for i in range(6)]) for row in range(y_true.shape[0])])
        max_value_s = np.sum([np.sum([y_true[row, i] ** 2 for i in range(6)]) for row in range(y_true.shape[0])])
        min_value = 0

        errors_l1 = [np.sum([np.abs(y_true[row, column] - y_pred[row, column]) for column in range(6)]) for row in range(y_true.shape[0])]
        errors_exp = [np.sum([np.exp(np.abs(y_true[row, column] - y_pred[row, column])) for column in range(6)]) for row in range(y_true.shape[0])]
        errors_squared = [np.sum([(y_true[row, column] - y_pred[row, column]) ** 2 for column in range(6)]) for row in range(y_true.shape[0])]

        # ERRORS COMPUTATION:
        # distance L1 between predictions and true values for each area
        Normalized_error_l1 = (np.sum(errors_l1) - min_value) / (max_value_l1 - min_value)
        # distance L2 between predictions and true values for each area
        Normalized_error_squared = (np.sum(errors_squared) - min_value) / (max_value_s - min_value)
        # distance exp between predictions and true values for each area
        Normalized_error_exp = (np.sum(errors_exp) - min_value) / (max_value_exp - min_value)
        # ACCURACY COMPUTATION:
        # accuracy distance exp between predictions and true values for each area
        accuracy_distance_exp = 1 - Normalized_error_exp
        # accuracy distance L1 between predictions and true values for each area
        accuracy_distance_l1 = 1 - Normalized_error_l1
        # accuracy distance L2 between predictions and true values for each area
        accuracy_distance_squared = 1 - Normalized_error_squared

        areas_dict = {'acc_a': 0.0,
                      'acc_b': 0.0,
                      'acc_c': 0.0,
                      'acc_d': 0.0,
                      'acc_e': 0.0,
                      'acc_f': 0.0}

        for i, area_key in zip(range(y_pred.shape[1]), areas_dict.keys()):
            areas_dict[area_key] = 100 * (np.sum(y_true[:, i] == y_pred[:, i]) / y_true.shape[0])
        y_pred = np.concatenate((y_pred, score_pred_RR.reshape(-1, 1), score_pred_LL.reshape(-1, 1), score_pred_G.reshape(-1, 1)), axis=1)
        return (accuracy_distance_l1,
                accuracy_distance_exp,
                accuracy_distance_squared,
                y_pred,
                {'acc_boolean_accuracy': accuracy_G,
                 'acc_lung_r': accuracy_RR,
                 'acc_lung_l': accuracy_LL,
                 **areas_dict})


def get_metrics_regression(y_true, y_pred, regression_type='area'):
    """
    This function computes the metrics for the regression task, MSE, L1, R2, explained variance, Correlation coefficient,
    and std deviation between predictions and true values
    """
    if regression_type == 'area':
        y_true = y_true[:, :6]
        # COMPUTE CORRELATION COEFFICIENTS:
        results = pd.DataFrame(index=['A', 'B', 'C', 'D', 'E', 'F', 'RL', 'LL', 'G'],
                               columns=['CC', 'SD', 'MSE', 'L1', 'R2'])

        for dim, area in zip(range(y_true.shape[1]), ['A', 'B', 'C', 'D', 'E', 'F']):
            corr_area, _ = pearsonr(y_true[:, dim], y_pred[:, dim])
            std_area = np.std(np.abs(y_true[:, dim] - y_pred[:, dim]))
            mse_area = mean_squared_error(y_true[:, dim], y_pred[:, dim])
            L1s_area = mean_absolute_error(y_true[:, dim], y_pred[:, dim])
            r2_area = r2_score(y_true[:, dim], y_pred[:, dim])
            results.loc[area, 'CC'] = np.round(corr_area, 4)
            results.loc[area, 'SD'] = np.round(std_area, 4)
            results.loc[area, 'MSE'] = np.round(mse_area, 4)
            results.loc[area, 'L1'] = np.round(L1s_area, 4)
            results.loc[area, 'R2'] = np.round(r2_area, 4)

        # REGIONS METRICS
        y_true_region_R = np.sum(y_true[:, :3], axis=1)
        y_true_region_L = np.sum(y_true[:, 3:6], axis=1)
        if regression_type == 'area':
            y_pred_region_R = np.sum(y_pred[:, :3], axis=1)
            y_pred_region_L = np.sum(y_pred[:, 3:6], axis=1)

        corr_RL, _ = pearsonr(y_true_region_R, y_pred_region_R)
        corr_LL, _ = pearsonr(y_true_region_L, y_pred_region_L)
        std_RL = np.std(np.abs(y_true_region_R - y_pred_region_R))
        std_LL = np.std(np.abs(y_true_region_L - y_pred_region_L))
        if regression_type == 'area':
            mse_RL = mean_squared_error(y_true[:, :3], y_pred[:, :3])
            mse_LL = mean_squared_error(y_true[:, 3:6], y_pred[:, 3:6])

            L1s_RL = mean_absolute_error(y_true[:, :3], y_pred[:, :3])
            L1s_LL = mean_absolute_error(y_true[:, 3:6], y_pred[:, 3:6])

            r2_RL = r2_score(y_true[:, :3], y_pred[:, :3])
            r2_LL = r2_score(y_true[:, 3:6], y_pred[:, 3:6])

        y_pred_global = np.sum(y_pred, axis=1)
        y_true_global = np.sum(y_true, axis=1)
        corr_global, _ = pearsonr(y_true_global, y_pred_global)
        std_global = np.std(np.abs(y_true_global - y_pred_global))
        mse_global = mean_squared_error(y_true[:, :], y_pred[:, :])
        L1s_global = mean_absolute_error(y_true[:, :], y_pred[:, :])
        r2_global = r2_score(y_true[:, :], y_pred[:, :])

        results.loc['RL', 'CC'] = np.round(corr_RL, 4)
        results.loc['RL', 'SD'] = np.round(std_RL, 4)
        results.loc['RL', 'MSE'] = np.round(mse_RL, 4)
        results.loc['RL', 'L1'] = np.round(L1s_RL, 4)
        results.loc['RL', 'R2'] = np.round(r2_RL, 4)

        results.loc['LL', 'CC'] = np.round(corr_LL, 4)
        results.loc['LL', 'SD'] = np.round(std_LL, 4)
        results.loc['LL', 'MSE'] = np.round(mse_LL, 4)
        results.loc['LL', 'L1'] = np.round(L1s_LL, 4)
        results.loc['LL', 'R2'] = np.round(r2_LL, 4)

        results.loc['G', 'CC'] = np.round(corr_global, 4)
        results.loc['G', 'SD'] = np.round(std_global, 4)
        results.loc['G', 'MSE'] = np.round(mse_global, 4)
        results.loc['G', 'L1'] = np.round(L1s_global, 4)
        results.loc['G', 'R2'] = np.round(r2_global, 4)
    elif regression_type == 'consistent':
        return None

    return results.T


def train_severity(model,
                   cfg,
                   criterion,
                   optimizer,
                   scheduler,
                   model_file_name,
                   dataloaders,
                   model_dir,
                   device,
                   num_epochs=25,
                   regression_type='area',
                   max_epochs_stop=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e6
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': [],
               'train_acc_G': [], 'val_acc_G': [], 'train_acc_RL': [], 'val_acc_RL': [], 'train_acc_LL': [], 'val_acc_LL': [],
               'train_l1_acc': []} if regression_type == 'area' else {'train_loss': [], 'val_loss': [], 'train_acc_G': [], 'val_acc_G': [],
                                                                      'train_acc_RL': [], 'val_acc_RL': [], 'train_acc_LL': [], 'val_acc_LL': []}

    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            output_predictions = None
            real_target = None

            # Iterate over data.
            number_samples = 0
            with tqdm(total=len(dataloaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}',
                      unit='img') as pbar:
                for k, outs in enumerate(dataloaders[phase]):
                    inputs = outs[0]
                    labels = outs[1]
                    file_name = outs[2]

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())

                        labels = labels.type(torch.float32)
                        # REMEMBER LABELS/OUTPUT ORDER = [A, B, C, D, E, F, RL, LL, G]

                        # This criterion calculate MSE over single areas: (MEAN-SQUARED ERROR)
                        if regression_type == 'area':
                            labels_single_areas = labels[:, :6]
                            loss = criterion(outputs, labels_single_areas)
                        elif regression_type == 'consistent':
                            # This criterion calculate a custom loss function
                            labels_consistent = labels
                            total_output = criterion(outputs, labels_consistent)
                            loss = total_output['total_loss']
                            labels_predicted = total_output['labels_predicted']
                            probs_predicted = total_output['probs_predicted']

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # Collect predictions and true labels
                    if regression_type == 'area':
                        if k == 0:
                            output_predictions = outputs.cpu().detach().numpy()
                            real_target = labels.cpu().detach().numpy()
                        else:
                            output_predictions = np.concatenate((output_predictions, outputs.cpu().detach().numpy()), 0)
                            real_target = np.concatenate((real_target, labels.cpu().detach().numpy()), 0)
                        # real_target = np.append(real_target, labels.cpu().detach().numpy())
                    elif regression_type == 'consistent':
                        if k == 0:
                            output_predictions = labels_predicted.cpu().detach().numpy()
                            real_target = labels.cpu().detach().numpy()
                            probs_predictions = probs_predicted.cpu().detach().numpy()
                        else:
                            output_predictions = np.concatenate((output_predictions, labels_predicted.cpu().detach().numpy()), 0)
                            real_target = np.concatenate((real_target, labels.cpu().detach().numpy()), 0)
                            probs_predictions = np.concatenate((probs_predictions, probs_predicted.cpu().detach().numpy()), 0)
                        # real_target = np.append(real_target, labels.cpu().detach().numpy())
                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    pbar.update(inputs.shape[0])

            metrics_epoch = get_metrics_regression(real_target, output_predictions, regression_type=regression_type)
            (_, _, _, _,
             acc_all_) = get_metrics_classification_severity(real_target, output_predictions, regression_type=regression_type)
            print('acc_all_', acc_all_)
            print('metrics_epoch', metrics_epoch)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'val':
                val_loss = epoch_loss
                update_learning_rate(optimizer=optimizer, scheduler=scheduler, metric=val_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                if regression_type == 'area':
                    history['train_metrics'].append(metrics_epoch)
                history['train_acc_G'].append(acc_all_['acc_boolean_accuracy'])
                history['train_acc_RL'].append(acc_all_['acc_lung_r'])
                history['train_acc_LL'].append(acc_all_['acc_lung_l'])

            else:
                history['val_loss'].append(epoch_loss)
                if regression_type == 'area':
                    history['val_metrics'].append(metrics_epoch)
                history['val_acc_G'].append(acc_all_['acc_boolean_accuracy'])
                history['val_acc_RL'].append(acc_all_['acc_lung_r'])
                history['val_acc_LL'].append(acc_all_['acc_lung_l'])
            # deep copy the model
            if epoch > cfg['trainer']['warmup_epochs']:
                if phase == 'val':
                    if epoch_loss < best_loss:
                        best_epoch = epoch
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(best_model_wts, os.path.join(model_dir, model_file_name + '.pt'))
    print('-----------------------------------'
          '\n Best Model Saved in: %s' % (os.path.join(model_dir, model_file_name)))

    return model, history


def train_MultiTask(model,
                    optimizer,
                    scheduler,
                    criterion,
                    model_file_name,
                    dataloaders,
                    model_dir,
                    device,
                    cfg,
                    num_epochs=25,
                    max_epochs_stop=3,
                    save_model=True):
    since = time.time()
    print('Noise added to the model')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_accuracy = 0.0
    best_loss = 1e6
    best_epoch = 0
    history = {'train_loss': [],
               'train_loss_S': [],
               'train_loss_M': [],
               'val_loss': [],
               'val_loss_S': [],
               'val_loss_M': [],
               'train_acc': [],
               'val_acc': []}
    epochs_no_improve = 0
    early_stop = False
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = {'Loss_M': 0.0, 'Loss_S': 0.0, 'Loss_TOT': 0.0}
            running_corrects = 0
            count_AFC = 0
            count_BX = 0
            # Iterate over data.
            with tqdm(total=len(dataloaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}',
                      unit='img') as pbar:
                for outs in dataloaders[phase]:
                    inputs = outs[0]
                    labels = outs[1]
                    file_name = outs[2]
                    dataset_class = outs[3]

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # Train updating of weights by gradient descent

                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(inputs.float())
                        # Outputs of the two heads

                        Severity_head_outputs = outputs[1]

                        labels = labels.type(torch.float32)
                        if cfg['model']['softmax']:
                            out_M = nn.Softmax(dim=1)(outputs[0])
                            outputs = (out_M, outputs[1])
                        Morbidity_head_outputs = outputs[0]
                        # Loss for multi-task
                        losses, selectors = criterion(outputs, labels, dataset_class=dataset_class)

                        # Selectors for data informations
                        BX_selector = selectors['BX_sel']
                        AFC_selector = selectors['AFC_sel']

                        # Labels/Outputs BX
                        labels_BX = labels[BX_selector]
                        outputs_BX = Severity_head_outputs[BX_selector]
                        # Labels/Outputs AFC
                        labels_AFC = labels[AFC_selector][:, :criterion.dim_1]
                        outputs_AFC = Morbidity_head_outputs[AFC_selector]

                        # Calculate predictions and Labels fgt for the Morbidity Task:
                        _, preds = torch.max(outputs_AFC, 1)
                        _, labels_gt = torch.max(labels_AFC, 1)

                        # Losses:
                        loss_AFC = losses['Loss_M']
                        loss_BX = losses['Loss_S']
                        loss = losses['Loss_TOT']

                        # Updates Counts:
                        count_AFC += labels_AFC.size(0)
                        count_BX += labels_BX.size(0)

                        pbar.set_postfix(**{'Total loss (batch)': loss.item(), 'AFC loss (batch)': loss_AFC.item(), 'BX loss (batch)': loss_BX.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics

                    running_loss['Loss_M'] += loss_AFC.item() * inputs.size(0)

                    running_loss['Loss_S'] += loss_BX.item() * inputs.size(0)

                    running_loss['Loss_TOT'] += loss.item() * inputs.size(0)

                    running_corrects += torch.sum(preds == labels_gt.data)

                    pbar.update(inputs.shape[0])

            epoch_loss_tot = running_loss['Loss_TOT'] / len(dataloaders[phase].dataset)
            epoch_loss_AFC = running_loss['Loss_M'] / (count_AFC + 1e-10)
            epoch_loss_BX = running_loss['Loss_S'] / (count_BX + 1e-10)

            if phase == 'val':
                val_loss = epoch_loss_tot
                update_learning_rate(optimizer=optimizer, scheduler=scheduler, metric=val_loss)
            epoch_acc = (running_corrects.double() / (count_AFC + 1e-10)).item()

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss_tot)
                history['train_loss_S'].append(epoch_loss_BX)
                history['train_loss_M'].append(epoch_loss_AFC)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss_tot)
                history['val_loss_S'].append(epoch_loss_BX)
                history['val_loss_M'].append(epoch_loss_AFC)
                history['val_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss_tot, epoch_acc))

            # deep copy the model
            if epoch > cfg['trainer']['warmup_epochs']:
                if phase == 'val':
                    if epoch_loss_AFC < best_loss:
                        best_epoch = epoch
                        best_epoch_accuracy = epoch_acc
                        best_loss = epoch_loss_AFC
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Acc: {:4f}'.format(best_epoch_accuracy))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if save_model:
        # Save model
        torch.save(best_model_wts, os.path.join(model_dir, model_file_name + '.pt'))
        print('-----------------------------------'
              '\n Best Model-MultiTask Saved in: %s' % (os.path.join(model_dir, model_file_name)))
        # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()



    return model, history, {'best_loss': best_loss, 'best_acc': best_epoch_accuracy, 'best_epoch': best_epoch}



def train_morbidity(model,
                    criterion,
                    optimizer,
                    scheduler,
                    model_file_name,
                    dataloaders,
                    model_dir,
                    device,
                    cfg,
                    num_epochs=25,
                    max_epochs_stop=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1e6
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(total=len(dataloaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}',
                      unit='img') as pbar:
                for outs in dataloaders[phase]:
                    inputs = outs[0]
                    labels = outs[1]
                    file_name = outs[2]

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # Train updating of weights by gradient descent

                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(inputs.float())
                        if cfg['model']['softmax']:
                            outputs = nn.Softmax(dim=1)(outputs)

                        _, preds = torch.max(outputs, 1)
                        _, labels_gt = torch.max(labels, 1)
                        loss = criterion(outputs, labels)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels_gt.data)

                    pbar.update(inputs.shape[0])

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == 'val':
                val_loss = epoch_loss
                update_learning_rate(optimizer=optimizer, scheduler=scheduler, metric=val_loss)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            elif phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch > cfg['trainer']['warmup_epochs']:
                if phase == 'val':
                    if epoch_loss < best_loss and epoch != 0:
                        best_epoch = epoch
                        best_acc = epoch_acc
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(f'\nEarly Stopping! Total epochs: {epoch}%')
                            early_stop = True
                            break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save weights model
    torch.save(best_model_wts, os.path.join(model_dir, model_file_name + '.pt'))
    print('-----------------------------------'
          '\n Best Model Saved in: %s' % (os.path.join(model_dir, model_file_name)))
    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def accuracy(output, target, topk=(1,)):
    """Compute the topk accuracy(s)"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        # For each k, find the percentage of correct
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def evaluate_multi_task(model, test_loader, criterion, cfg, device, idx_to_class_AFC, topk=(1, 5)):
    """Measure the performance of a trained MultiTask-MultiHead Model model"""

    # Final accuracy results: (S)
    S_preds = None
    S_target = None
    M_preds = []
    M_target = []
    print('-' * 10)
    model.eval()
    classes = []
    loss_BX_test = 0.0
    loss_AFC_test = []
    M_results_acc_temp = []
    # Each epoch has a training and validation phase
    with torch.no_grad():

        # Set model to evaluate mode
        count_AFC = 0
        count_BX = 0
        batch_elaborated = 0
        i = 0
        # Iterate over data.
        with tqdm(total=len(test_loader), desc=f'Testing MultiTask Model on dobule test-set (M + S)',
                  unit='img') as pbar:
            for inputs, labels, file_name, dataset_class in test_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward

                outputs = model(inputs.float())
                # Outputs of the two heads
                Morbidity_head_outputs = outputs[0]
                Severity_head_outputs = outputs[1]
                labels = labels.type(torch.float32)
                if cfg['model']['softmax']:
                    outputs = nn.Softmax(dim=1)(outputs)
                # Loss for multi-task
                losses, selectors = criterion(outputs, labels, dataset_class=dataset_class)
                # Selectors for data informations
                BX_selector = selectors['BX_sel']
                AFC_selector = selectors['AFC_sel']
                # Labels/Outputs BX
                labels_BX = labels[BX_selector]
                outputs_BX = Severity_head_outputs[BX_selector]
                # Labels/Outputs AFC
                labels_AFC = labels[AFC_selector][:, :criterion.dim_1]
                outputs_AFC = Morbidity_head_outputs[AFC_selector]
                # Calculate predictions and Labels for the Morbidity Task:
                _, preds = torch.max(outputs_AFC, 1)
                _, labels_gt = torch.max(labels_AFC, 1)
                # Losses:
                loss_AFC = losses['Loss_M']
                loss_BX = losses['Loss_S']
                # Updates Counts:
                count_AFC += labels_AFC.size(0)
                count_BX += labels_BX.size(0)

                # BX aggregation
                if batch_elaborated == 0:
                    S_preds = outputs_BX.cpu().detach().numpy()
                    S_target = labels_BX.cpu().detach().numpy()
                else:
                    S_preds = np.concatenate((S_preds, outputs_BX.cpu().detach().numpy()), 0)
                    S_target = np.concatenate((S_target, labels_BX.cpu().detach().numpy()), 0)
                loss_BX_test += loss_BX.item() * inputs.size(0)
                batch_elaborated += 1

                # AFC aggregation
                M_target.append(labels_gt.data.cpu().numpy())
                M_preds.append(preds.data.cpu().numpy())

                # Iterate through each example

                for pred, true, target in zip(outputs_AFC, labels_gt, labels_AFC):
                    # Find topk accuracy
                    M_results_acc_temp.append(accuracy(pred.unsqueeze(0), true.unsqueeze(0), topk))
                    classes.append(true.item())
                    loss_AFC_test.append(loss_AFC.item())
                    i += 1
    # AFC RESUME
    # Hold accuracy results
    M_results_acc = np.zeros((count_AFC, len(topk)))
    for i, value in enumerate(M_results_acc_temp):
        M_results_acc[i, :] = value

    M_true_labels = list(itertools.chain(*M_target))
    M_predicted_labels = list(itertools.chain(*M_preds))
    M_precision = precision_score(M_true_labels, M_predicted_labels)
    M_recall = recall_score(M_true_labels, M_predicted_labels)
    M_f1 = f1_score(M_true_labels, M_predicted_labels)
    M_roc_auc = roc_auc_score(M_true_labels, M_predicted_labels) if np.unique(M_true_labels).__len__() > 1 else 0.0
    # Send results to a dataframe and calculate average across classes
    results_morbidity = pd.DataFrame(M_results_acc, columns=[f'top{i}' for i in topk])
    results_morbidity['class'] = classes

    results_morbidity = results_morbidity.groupby(classes).mean()
    results_morbidity['class'] = results_morbidity['class'].apply(lambda x: idx_to_class_AFC[x])

    # BX RESUME
    loss_test_severity = losses
    metrics_severity = get_metrics_regression(S_target[:, :6], S_preds)

    M_acc = M_results_acc.mean()
    return results_morbidity, {
        "Accuracy": M_acc,
        "Precision": M_precision,
        "Recall": M_recall,
        "F1 Score": M_f1,
        "ROC AUC Score": M_roc_auc
    }, loss_test_severity, metrics_severity


def evaluate_regression(model, test_loader, criterion, device, cfg, regression_type='area'):
    """Measure the performance of a trained PyTorch model on a regression task
    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader):  dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category
    """
    losses = 0.0
    output_predictions = None
    real_target = None
    model.eval()
    with torch.no_grad():

        # ing loop

        # Iterate over data.
        number_samples = 0
        with tqdm(test_loader, total=len(test_loader), desc=f'Testing Regression Model',
                  unit='img') as pbar:
            k = 0
            filenames = []

            for outs in pbar:
                inputs = outs[0]
                labels = outs[1]
                file_name = outs[2]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward
                # track history if only in train
                outputs = model(inputs.float())
                labels = labels.type(torch.float32)
                # All the filenames
                filenames.extend([os.path.basename(file_).split('.dcm')[0] for file_ in file_name])
                if isinstance(outputs, tuple):
                    outputs = outputs[1]
                # This criterion calculate MSE over single areas: (MEAN-SQUARED ERROR)
                if regression_type == 'area':
                    labels_single_areas = labels[:, :6]
                    loss = criterion(outputs, labels_single_areas)
                elif regression_type == 'consistent':
                    labels_consistent = labels
                    total_output = criterion(outputs, labels_consistent)
                    loss = total_output['total_loss']
                    labels_predicted = total_output['labels_predicted']
                    probs_predicted = total_output['probs_predicted']
                losses += loss.item() * inputs.size(0)
                if regression_type == 'area':
                    if k == 0:
                        output_predictions = outputs.cpu().detach().numpy()
                        real_target = labels.cpu().detach().numpy()
                    else:
                        output_predictions = np.concatenate((output_predictions, outputs.cpu().detach().numpy()), 0)
                        real_target = np.concatenate((real_target, labels.cpu().detach().numpy()), 0)
                    # real_target = np.append(real_target, labels.cpu().detach().numpy())
                elif regression_type == 'consistent':
                    if k == 0:
                        output_predictions = labels_predicted.cpu().detach().numpy()
                        real_target = labels.cpu().detach().numpy()
                        probs_predictions = probs_predicted.cpu().detach().numpy()
                    else:
                        output_predictions = np.concatenate((output_predictions, labels_predicted.cpu().detach().numpy()), 0)
                        real_target = np.concatenate((real_target, labels.cpu().detach().numpy()), 0)
                        probs_predictions = np.concatenate((probs_predictions, probs_predicted.cpu().detach().numpy()), 0)
                k += 1

    # Calculate LOSS + Metrics
    loss_ = losses / len(test_loader.dataset)
    metrics_ = get_metrics_regression(real_target, output_predictions, regression_type=regression_type)
    (accuracy_distance_l1,
     accuracy_distance_exp,
     accuracy_distance_squared,
     y_sampling_pred,
     acc_all_) = get_metrics_classification_severity(real_target, output_predictions, regression_type=regression_type)
    if regression_type == 'area':
        # Save performances results
        metrics_to_add_CC = {keys + '_CC': values for keys, values in metrics_.loc['CC', ['LL', 'RL', 'G']].to_dict().items()}
        metrics_to_add_L1 = {keys + '_L1': values for keys, values in metrics_.loc['L1', ['LL', 'RL', 'G']].to_dict().items()}

        results_metrics_resume = {
            'Accuracy L1': accuracy_distance_l1,

            'Accuracy Exp': accuracy_distance_exp,

            'Accuracy Squared': accuracy_distance_squared,

            'Acc_G': acc_all_['acc_boolean_accuracy'],

            'Acc_LR': acc_all_['acc_lung_r'],

            'Acc_LL': acc_all_['acc_lung_l'],

            **metrics_to_add_L1,
            **metrics_to_add_CC
        }

        # Prediction by patient's image with floating values
        results_float = {'A_pred': output_predictions[:, 0],
                         'B_pred': output_predictions[:, 1],
                         'C_pred': output_predictions[:, 2],
                         'D_pred': output_predictions[:, 3],
                         'E_pred': output_predictions[:, 4],
                         'F_pred': output_predictions[:, 5],
                         'LR_pred': y_sampling_pred[:, -3],
                         'LL_pred': y_sampling_pred[:, -2],
                         'G_pred': y_sampling_pred[:, -1]}
    elif regression_type == 'consistent':
        results_float = {'A_pred': output_predictions[:, 0],
                         'B_pred': output_predictions[:, 1],
                         'C_pred': output_predictions[:, 2],
                         'D_pred': output_predictions[:, 3],
                         'E_pred': output_predictions[:, 4],
                         'F_pred': output_predictions[:, 5],
                         'LR_pred': output_predictions[:, :3].sum(1),
                         'LL_pred': output_predictions[:, :3].sum(1),
                         'G_pred': output_predictions[:, :].sum(1)}


        results_metrics_resume = {
            'Accuracy L1': accuracy_distance_l1,
            'Accuracy Exp': accuracy_distance_exp,
            'Accuracy Squared': accuracy_distance_squared,

            'Acc_G': acc_all_['acc_boolean_accuracy'],

            'Acc_LR': acc_all_['acc_lung_r'],

            'Acc_LL': acc_all_['acc_lung_l'],
            'Acc_A': acc_all_['acc_a'],
            'Acc_B': acc_all_['acc_b'],
            'Acc_C': acc_all_['acc_c'],
            'Acc_D': acc_all_['acc_d'],
            'Acc_E': acc_all_['acc_e'],
            'Acc_F': acc_all_['acc_f'],
        }

    # Lets save each zone with a name
    labels = {key.split('_')[0] + 'y': list() for key in results_float.keys()}
    for row in range(real_target.shape[0]):
        for col in zip(range(real_target.shape[1]), labels.keys()):
            labels[col[1]].append(real_target[row, col[0]])

    # ARRAY ALL
    labels = {key: np.array(values) for key, values in labels.items()}

    # COMMON KEYS
    list_common_keys = list(results_float.keys()) + list(labels.keys())
    # We have to alternate all the columns name and save all the information about all the prediction of the network
    results_for_images = pd.DataFrame(columns=['Patient'] + list(itertools.chain(*[[list_common_keys[i], list_common_keys[i + 9]] for i in range(9)])))

    new_info = pd.DataFrame({'Patient': filenames, **results_float, **labels})
    for row in new_info.iterrows():
        results_for_images.loc[len(results_for_images)] = dict(row[1])

    return metrics_, loss_, results_metrics_resume, results_for_images


def evaluate_morbidity(model, test_loader, criterion, idx_to_class, device, cfg, topk=(1, 5)):
    """Measure the performance of a trained PyTorch model
    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category
    """
    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros((len(test_loader.dataset), len(topk)))
    i = 0
    predicted_probs = []
    true_labels = []
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        results_for_images = pd.DataFrame(columns=['Patient', 'Predicted', 'labels', 'probs'])
        # Testing loop
        for outs in tqdm(test_loader):
            data = outs[0]
            targets = outs[1]
            file_name = outs[2]
            data = data.to(device)
            targets = targets.to(device)
            # Raw model output
            outputs = model(data.float())
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                targets = targets[:, :2]
            if cfg['model']['softmax']:
                outputs = nn.Softmax(dim=1)(outputs)

            values_probs, preds = torch.max(outputs, 1)
            _, labels_gt = torch.max(targets, 1)

            true_labels.append(labels_gt.data.cpu().numpy())
            predicted_labels.append(preds.data.cpu().numpy())

            # Iterate through each example
            new_info = pd.DataFrame({'Patient': file_name, 'Predicted': preds.data.cpu().numpy(), 'labels': labels_gt.data.cpu().numpy(), 'probs': values_probs.cpu().numpy()})
            for row in new_info.iterrows():
                results_for_images.loc[len(results_for_images)] = dict(row[1])

            for pred, true, target in zip(outputs, labels_gt, targets):
                # Find topk accuracy
                acc_results[i, :] = accuracy(pred.unsqueeze(0), true.unsqueeze(0), topk)
                classes.append(true.item())
                # Calculate the loss
                loss = criterion(pred, target)
                losses.append(loss.item())
                i += 1

    # Calculate metrics # we put the label to 1 if the second neuron is giving high porbability, or 0 in the other case

    true_labels = list(itertools.chain(*true_labels))
    predicted_labels = list(itertools.chain(*predicted_labels))

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels) if np.unique(true_labels).__len__() > 1 else 0.0
    # Send results to a dataframe and calculate average across classes
    results_classes = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    results_classes['class'] = classes
    results_classes['loss'] = np.mean(losses)

    results_classes = results_classes.groupby(classes).mean()
    results_classes['class'] = results_classes['class'].apply(lambda x: idx_to_class[x])
    acc = acc_results.mean()
    return results_for_images, results_classes, {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC Score": roc_auc
    }


def masked_pred(img, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [0, 0, 1]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


MODELS = {
    'ChestX-ray14': ['https://zenodo.org/record/5205234/files/chestxray14.pth.tar?download=1', 'pth'],
    'ChexPert': ['https://zenodo.org/record/5205234/files/chexpert.pth.tar?download=1', 'pth'],
    'ImageNet_ChestX-ray14': ['https://zenodo.org/record/5205234/files/ImageNet_chestxray14.pth.tar?download=1', 'pth'],
    'ImageNet_ChexPert': ['https://zenodo.org/record/5205234/files/ImageNet_chexpert.pth.tar?download=1', 'pth'],
}


def remove_keys(d):
    for key in list(d.keys()):
        if 'module.jigsaw' in key or 'module.head_jig' in key:
            print('warning, jigsaw stream in model')
            d.pop(key)
        elif 'projection' in key or 'prototypes' in key or 'linear' in key or 'head' in key:
            print(f'removed {key}')
            d.pop(key)
    return d


def rename(d, model_name):
    unwanted_prefixes = {
        'ChestX-ray14': '',
        'ChexPert': 'module.',
        'ImageNet_ChestX-ray14': '',
        'ImageNet_ChexPert': 'module.',
    }

    prefix = unwanted_prefixes[model_name]
    l = len(prefix)
    new_d = {}
    for key in d.keys():
        if prefix in key:
            new_d[key[l:]] = d[key]
        else:
            new_d[key] = d[key]
    return new_d


def load_resnet50(model_name, weights_dir='./weights'):
    path = os.path.join(weights_dir, f'{model_name}.pth.tar')
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    print(f"state_dict={state_dict.keys()}")

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'resnet' in state_dict:
        state_dict = state_dict['resnet']

    state_dict = rename(state_dict, model_name)
    state_dict = remove_keys(state_dict)

    return state_dict


def load_Chest_resnet_weights(chestXnet_model='ChestX-ray14'):
    # Weights directory
    weights_dir = './weights'
    # ChestXNet model pretrained weights
    model_pretrained_weights = os.path.join(weights_dir, chestXnet_model)
    if os.path.exists(model_pretrained_weights):
        return load_resnet50(chestXnet_model, weights_dir=model_pretrained_weights)
    else:
        assert AttributeError(f'No weights for {chestXnet_model} found in {model_pretrained_weights}')


class MorbidityModel(nn.Module):

    def __init__(self, cfg=None, backbone='resnet18', device=None, freezing_layer='', *args, **kwargs):

        super(MorbidityModel, self).__init__()
        layers_backbone = {'last': 2, 'last-1': 3}
        self.config = cfg
        self.device = device
        self.classes = cfg['data']['classes']
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        # BACKBONE
        self.backbone, in_features = get_backbone(backbone)
        self.softmax = nn.Softmax(dim=1)
        #
        New_classification_Head = nn.Sequential(OrderedDict(
            [('dropout1', nn.Dropout(p=0.5, inplace=False)),
             ('classification-Head',
              nn.Linear(in_features=in_features, out_features=len(self.classes)))]))
        # HEAD CLASSIFICATION
        if 'squeezenet' in backbone:
            New_classification_Head = nn.Sequential(OrderedDict(
                [
                    ('dropout2', nn.Dropout(p=0.5, inplace=False)),
                    ('classification-Head', nn.Conv2d(512, len(self.classes), kernel_size=(1, 1), stride=(1, 1)))
                ]
            ))
        if 'resnet50_ChexPert' in backbone or 'resnet50_ChestX-ray14' in backbone or 'resnet50_ImageNet_ChestX-ray14' in backbone or 'resnet50_ImageNet_ChexPert' in backbone:
            state_dict = load_Chest_resnet_weights(chestXnet_model=backbone.split('resnet50_')[1])
            print(f'Loading pretrained model {backbone}')
            self.backbone.load_state_dict(state_dict, strict=False)
        elif 'densenet121_CXR' in backbone:
            # load Checkpoint
            path_to_chestXnet_model = './weights/chestXnet/checkpoint'
            checkpoint = torch.load(path_to_chestXnet_model, map_location=lambda storage, loc: storage)
            model = checkpoint['model']

            prefix = 'classifier.'
            adapted_dict = {k: v for k, v in model.state_dict().items()
                            if not k.startswith(prefix)}

            print('Loading pretrained model on ChestXNet, from: ', path_to_chestXnet_model)
            self.backbone.load_state_dict(adapted_dict, strict=False)

        # CHANGE HEAD
        for i, param in enumerate(list(New_classification_Head.parameters())):
            param.requires_grad = True

        # (AIFORCOVID) MORBIDITY HEAD
        change_head(model=self.backbone, model_name=backbone, new_head=New_classification_Head)

        # Freeze Backbone
        if self.config['model']['freezing']:
            freeze_backbone(model=self, perc=0.70)

    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            pass


class SeverityModel(nn.Module):

    def __init__(self, cfg=None, backbone='resnet18', device=None, freezing_layer='', *args, **kwargs):

        super(SeverityModel, self).__init__()
        layers_backbone = {'last': 2, 'last-1': 3}
        self.config = cfg
        self.device = device
        self.labels = cfg['data']['classes']
        self.class_to_id = {c: i for i, c in enumerate(self.labels)}
        # BACKBONE
        self.backbone, in_features = get_backbone(backbone)
        # HEAD CLASSIFICATION
        New_classification_Head = nn.Sequential(OrderedDict(
            [
                ('dropout1', nn.Dropout(p=0.5, inplace=False)),
                ('classification-Head',
                 nn.Linear(in_features=in_features, out_features=len(self.labels)))]))
        # HEAD CLASSIFICATION
        if 'squeezenet' in backbone:
            New_classification_Head = nn.Sequential(OrderedDict(
                [
                    ('dropout1', nn.Dropout(p=0.5, inplace=False)),
                    ('classification-Head', nn.Conv2d(512, len(self.labels), kernel_size=(1, 1), stride=(1, 1)))
                ]
            ))
        if 'resnet50_ChexPert' in backbone or 'resnet50_ChestX-ray14' in backbone or 'resnet50_ImageNet_ChestX-ray14' in backbone or 'resnet50_ImageNet_ChexPert' in backbone:
            state_dict = load_Chest_resnet_weights(chestXnet_model=backbone.split('resnet50_')[1])
            print(f'Loading pretrained model {backbone}')
            self.backbone.load_state_dict(state_dict, strict=False)
        elif 'densenet121_CXR' in backbone:
            # load Checkpoint
            path_to_chestXnet_model = './weights/chestXnet/checkpoint'
            checkpoint = torch.load(path_to_chestXnet_model, map_location=lambda storage, loc: storage)
            model = checkpoint['model']

            prefix = 'classifier.'
            adapted_dict = {k: v for k, v in model.state_dict().items()
                            if not k.startswith(prefix)}

            print('Loading pretrained model on ChestXNet, from: ', path_to_chestXnet_model)
            self.backbone.load_state_dict(adapted_dict, strict=False)
        if cfg['model']['structure'] == 'brixia':
            self.global_ = nn.Sequential(OrderedDict(
                [('dropout_global', nn.Dropout(p=0.5, inplace=False)),
                 ('relu_global', nn.ReLU()),
                 ('linear_global', nn.Linear(in_features=in_features, out_features=128))]))
            self.area_A_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_A', nn.ReLU()),
                    ('area_A',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_B_output = nn.Sequential(OrderedDict(
                [('ReLU_B', nn.ReLU()),
                 ('area_B',
                  nn.Linear(in_features=128, out_features=4))]))
            self.area_C_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_C', nn.ReLU()),
                    ('area_C',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_D_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_D', nn.ReLU()),
                    ('area_D',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_E_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_E', nn.ReLU()),
                    ('area_E',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_F_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_F', nn.ReLU()),
                    ('area_F',
                     nn.Linear(in_features=128, out_features=4))]))

            change_head(model=self.backbone, model_name=backbone, new_head=self.global_)
            for i, param in enumerate(list(self.backbone.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_A_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_B_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_C_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_D_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_E_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_F_output.parameters())):
                param.requires_grad = True

        else:
            change_head(model=self.backbone, model_name=backbone, new_head=New_classification_Head)
            for i, param in enumerate(list(New_classification_Head.parameters())):
                param.requires_grad = True
        """
        # (AIFORCOVID) MORBIDITY HEAD
        if False:
            freeze_backbone(model=self, perc=0.5)"""

    def forward(self, x):
        if self.config['model']['structure'] == 'brixia':

            # Areas
            out_global_memory = self.backbone(x)
            out_area_A = self.area_A_output(out_global_memory)[:, None, :]
            out_area_B = self.area_B_output(out_global_memory)[:, None, :]
            out_area_C = self.area_C_output(out_global_memory)[:, None, :]
            out_area_D = self.area_D_output(out_global_memory)[:, None, :]
            out_area_E = self.area_E_output(out_global_memory)[:, None, :]
            out_area_F = self.area_F_output(out_global_memory)[:, None, :]
            return torch.cat((out_area_A, out_area_B, out_area_C, out_area_D, out_area_E, out_area_F), dim=1)
        else:
            x = self.backbone(x)
            return x


def get_SingleTaskModel(kind='morbidity', backbone='', cfg=None, device=None, *args, **kwargs):
    if cfg['model']['task'] == 'morbidity':
        return MorbidityModel(cfg=cfg, backbone=backbone, device=device, *args, **kwargs)
    if cfg['model']['task'] == 'severity':
        return SeverityModel(cfg=cfg, backbone=backbone, device=device, *args, **kwargs)


def get_MultiTaskModel(kind='parallel', backbone='', cfg=None, device=None, *args, **kwargs):
    # TODO Multitask model
    if kind == 'parallel':
        return ParallelMultiObjective(cfg=cfg, backbone=backbone, device=device, *args, **kwargs)
    if kind == 'serial':
        return SerialMultiObjective(cfg=cfg, backbone=backbone, device=device, *args, **kwargs)


def MSE_loss(outputs, labels):
    number = (len(outputs) - len(labels[torch.all(torch.isnan(outputs), dim=1)])) if (len(outputs) - len(labels[torch.all(torch.isnan(outputs), dim=1)])) != 0 else 1

    return torch.nansum((outputs - labels) ** 2) / number


class BrixiaCustomLoss(torch.nn.Module):
    def __init__(self, cfg: dict, eps: float = 1e-08):
        super(BrixiaCustomLoss, self).__init__()
        self.cfg = cfg
        self.eps = eps
        self.alpha = cfg['trainer']['alpha']
        self.C = 4
        self.one_hot = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}
        self.reverse_one_hot = {tuple(v): k for k, v in self.one_hot.items()}

    def get_one_hot(self, x):
        output_tensor = torch.zeros([x.shape[0], 4])
        for value in range(x.shape[0]):
            one_hotted = self.one_hot[x[value].item()]
            output_tensor[value, :] = torch.Tensor(one_hotted)
        return output_tensor

    def forward(self, outputs, labels, **kwargs):
        # REMEMBER LABELS/OUTPUT ORDER = [A, B, C, D, E, F, RL, LL, G]
        labels_areas = labels[:, :6]
        outputs_areas = outputs[:, :]

        loss_total_BCE_categorical = 0
        loss_total_regression = 0

        labels_predicted = torch.zeros_like(labels_areas)
        probs_predicted = torch.zeros_like(labels_areas)

        for s_area in range(labels_areas.shape[-1]):

            labels_score = labels_areas[:, s_area].to(outputs.device)
            labels_encoded = self.get_one_hot(labels_areas[:, s_area]).to(outputs.device)
            outputs_area = outputs_areas[:, s_area, :].to(outputs.device)

            loss_total_BCE_categorical += torch.mean(nn.BCEWithLogitsLoss(reduction="none")(outputs_area, labels_encoded).mean(dim=1))
            out_S = nn.Softmax(dim=1)(outputs_area).to(outputs.device)
            probs, label = torch.max(out_S, dim=1)
            # add results:
            labels_predicted[:, s_area] = label
            probs_predicted[:, s_area] = probs

            factors = torch.zeros_like(out_S)
            for i in range(factors.shape[0]):
                factors[i, :] = torch.Tensor([0, 1, 2, 3])
            # Regression OUTPUT
            output_regressed = torch.mul(out_S, factors.to(outputs.device)).sum(dim=1)
            loss_total_regression += torch.mean(nn.L1Loss(reduction="none")(output_regressed.to(outputs.device), labels_score))

        # Total loss
        total_loss = self.alpha * loss_total_BCE_categorical + (1 - self.alpha) * loss_total_regression
        return {'total_loss': total_loss, 'labels_predicted': labels_predicted, 'probs_predicted': probs_predicted, 'regression_output': output_regressed}


class IdentityMultiHeadLoss(torch.nn.Module):
    def __init__(self, cfg: dict, eps: float = 1e-08):
        super(IdentityMultiHeadLoss, self).__init__()
        self.cfg = cfg
        self.regression = self.cfg.model.regression_type
        self.loss_1 = cfg.trainer.loss_1
        self.loss_2 = cfg.trainer.loss_2
        self.encode_dataset = {'AFC': 0, 'BX': 1}
        self.dim_1 = 2
        encode_bx_dim = {'area': 6, 'region': 3, 'global': 1, 'consistent': 24}
        self.dim_2 = encode_bx_dim[self.regression]

    def get_brixia_mask(self, labels, dim=6):
        batch_dim = labels.shape[0]

        mask = torch.zeros((batch_dim, dim))
        index = [(label == 1).item() for label in labels]
        mask[index] = torch.ones(1, dim)
        return mask

    def get_aiforcovid_mask(self, labels, dim=2):
        batch_dim = labels.shape[0]

        labels = torch.abs(labels - 1)
        mask = torch.zeros((batch_dim, dim))

        index = [(label == 1).item() for label in labels]

        mask[index] = torch.ones(1, dim)

        return mask

    def encode_batch(self, labels_class):
        return [self.encode_dataset[label] for label in labels_class]

    def get_loss(self, name):
        if name.upper() == 'MSE':
            return nn.MSELoss(reduction='sum')
        elif name.upper() == 'BCE':
            return nn.BCELoss(reduction='sum')
        elif name.upper() == 'CE':
            return nn.CrossEntropyLoss(reduction='sum')
        elif name.upper() == 'BRIXIA':
            return BrixiaCustomLoss(cfg=self.cfg)
        else:
            raise NotImplementedError

    def forward(self, outputs, labels, **kwargs):

        labels = labels[:, :6]
        labels_class = kwargs['dataset_class']
        # Create a new vector with the labels that link each sample to the corresponding dataset
        labels_class = torch.Tensor(self.encode_batch(labels_class)).to(labels.device)
        BX_mask = self.get_brixia_mask(labels=labels_class, dim=self.dim_2).to(labels.device)
        if self.cfg['model']['structure_bx'] == 'brixia':
            BX_mask_output = BX_mask.view(BX_mask.shape[0], 6, 4)
        AFC_mask = self.get_aiforcovid_mask(labels=labels_class, dim=self.dim_1).to(labels.device)

        BX_selector = ~torch.any(BX_mask == 0, dim=1)
        AFC_selector = ~torch.any(AFC_mask == 0, dim=1)

        # Calculate the loss function for each head/Task
        Loss_AFC = self.get_loss(self.loss_1)
        Loss_BX = self.get_loss(self.loss_2)

        # Calculate the loss for each head
        loss_1 = Loss_AFC(outputs[0] * AFC_mask, labels[:, :self.dim_1] * AFC_mask)

        loss_2 = Loss_BX(outputs[1] * BX_mask_output, labels[:, :6] * BX_mask[:, :6])['total_loss'] if self.cfg['model']['structure_bx'] == 'brixia' else Loss_BX(outputs[1] * BX_mask,
                                                                                                                                                                  labels[:, :6] * BX_mask)
        number_AFC = torch.count_nonzero(AFC_selector) if torch.count_nonzero(AFC_selector).item() != 0 else 1
        number_BX = torch.count_nonzero(BX_selector) if torch.count_nonzero(BX_selector).item() != 0 else 1
        # Calculate the total loss
        Loss_TOT =  torch.div(loss_1, number_AFC)  +  torch.mul(torch.div(loss_2, number_BX), 1/5)
        return {'Loss_TOT': Loss_TOT, 'Loss_M': loss_1, 'Loss_S': loss_2}, {'AFC_sel': AFC_selector, 'BX_sel': BX_selector}



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            device = m.weight.device
            noise = torch.randn(m.weight.size(), device=device) * 0.03
            m.weight.add_(noise)


class ParallelMultiObjective(nn.Module):
    def __init__(self, cfg=None, backbone='resnet18', device=None, *args, **kwargs):
        super(ParallelMultiObjective, self).__init__()
        self.config = cfg
        self.device = device

        self.cfg_morbidity = cfg['data']['modes']['morbidity']
        self.cfg_severity = cfg['data']['modes']['severity']
        self.classes_morbidity = self.cfg_morbidity['classes']
        self.classes_severity = self.cfg_severity['classes']

        self.labels = [*self.cfg_morbidity['classes'], *self.cfg_severity['classes']]
        self.class_to_id = {c: i for i, c in enumerate(self.labels)}
        # BACKBONE
        self.softmax = nn.Softmax(dim=1)
        self.backbone, self.in_features = get_backbone(backbone)
        self.drop_rate = self.config['model']['dropout_rate']
        self.train_backbone = self.config['model']['train_backbone']

        # HEAD CLASSIFICATION : MORBIDITY
        # HEAD CLASSIFICATION : MORBIDITY

        if cfg['model']['structure_bx'] == 'brixia':
            self.global_ = nn.Sequential(OrderedDict(
                [('dropout_global', nn.Dropout(p=0.5, inplace=False)),
                 ('relu_global', nn.ReLU()),
                 ('linear_global', nn.Linear(in_features=self.in_features, out_features=128))]))
            self.area_A_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_A', nn.ReLU()),
                    ('area_A',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_B_output = nn.Sequential(OrderedDict(
                [('ReLU_B', nn.ReLU()),
                 ('area_B',
                  nn.Linear(in_features=128, out_features=4))]))
            self.area_C_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_C', nn.ReLU()),
                    ('area_C',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_D_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_D', nn.ReLU()),
                    ('area_D',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_E_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_E', nn.ReLU()),
                    ('area_E',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_F_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_F', nn.ReLU()),
                    ('area_F',
                     nn.Linear(in_features=128, out_features=4))]))

            for i, param in enumerate(list(self.backbone.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_A_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_B_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_C_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_D_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_E_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_F_output.parameters())):
                param.requires_grad = True
        else:
            print('HEAD CLASSIFICATION :', self.config['model']['head'])

            self.Head_Severity = nn.Sequential(
                OrderedDict([
                    ('S_linear0', nn.Linear(in_features=self.in_features, out_features=128, bias=True)),
                    ('S_ReLU0', nn.ReLU()),
                    ('S_Dropout0', nn.Dropout(p=self.drop_rate)),
                    ('S_linear1', nn.Linear(in_features=128, out_features=32, bias=True)),
                    ('S_ReLU1', nn.ReLU()),
                    ('S_Dropout1', nn.Dropout(p=self.drop_rate)),
                    ('S_classification-Head', nn.Linear(in_features=32, out_features=len(self.classes_severity), bias=True))
                ]))

            init_weights(self.Head_Severity)
        self.Head_Morbidity = nn.Sequential(
            OrderedDict([
                ('M_linear0', nn.Linear(in_features=self.in_features, out_features=128, bias=True)),
                ('M_ReLU0', nn.ReLU()),
                ('M_Dropout0', nn.Dropout(p=self.drop_rate)),
                ('M_linear1', nn.Linear(in_features=128, out_features=32, bias=True)),
                ('M_ReLU2', nn.ReLU()),
                ('M_Dropout1', nn.Dropout(p=self.drop_rate)),
                ('M_classification-Head', nn.Linear(in_features=32, out_features=len(self.classes_morbidity), bias=True))
            ]))
        init_weights(self.Head_Morbidity)
        # Turn on the training of the gradients
        change_head(model=self.backbone, model_name=backbone, new_head=nn.Identity())

        # Freeze Backbone
        if self.config['model']['freezing']:
            freeze_backbone(model=self, perc=0.70)

    def activate_Head_training_module(self):
        for head in [self.Head_Morbidity, self.Head_Severity]:
            for i, param in enumerate(list(head.parameters())):
                param.requires_grad = True
        for i, param in enumerate(list(self.backbone.parameters())):
            param.requires_grad = self.train_backbone

    def load_backbone_average_weights(self, morbidity_params, severity_params, Beta=0.):
        """
        This function load the mean weights of the two models, and fuse them together with an adaptive mean. If Beta == 0, the weights
        loaded in the model are coming from only the severity model, if Beta == 1, the weights loaded in the model are coming from only the morbidity model.

        :param morbidity_params: Morbidity model parameters
        :param severity_params: Severity model parameters
        :param Beta: Beta parameter for the weighted mean
        :return: None
        """
        # TODO Adapt this one ?
        print('LOADING BACKBONE WEIGHTS (1/2 * S + 1/2 * M)')
        this_model_params = dict(self.named_parameters())
        for (name_m, param_morbidity), (name_s, param_severity) in zip(morbidity_params.items(), severity_params.items()):
            if 'module' in name_m:
                name_m = name_m.split('module.')[1]
            if 'module' in name_s:
                name_s = name_s.split('module.')[1]
            if name_m == name_s:

                if name_m in this_model_params and name_s in this_model_params:
                    this_model_params[name_m].data.copy_((Beta * param_morbidity.data + (1 - Beta) * param_severity.data))
        self.load_state_dict(this_model_params, strict=False)

    def forward(self, x):
        x_hat = self.backbone(x)
        if self.config['model']['structure_bx'] == 'brixia':
            # Areas and Global scorer
            out_global_memory = self.global_(x_hat)
            out_area_A = self.area_A_output(out_global_memory)[:, None, :]
            out_area_B = self.area_B_output(out_global_memory)[:, None, :]
            out_area_C = self.area_C_output(out_global_memory)[:, None, :]
            out_area_D = self.area_D_output(out_global_memory)[:, None, :]
            out_area_E = self.area_E_output(out_global_memory)[:, None, :]
            out_area_F = self.area_F_output(out_global_memory)[:, None, :]
            out_severity = torch.cat((out_area_A, out_area_B, out_area_C, out_area_D, out_area_E, out_area_F), dim=1)
        else:
            out_severity = self.Head_Severity(x_hat)
        out_morbidity = self.softmax(self.Head_Morbidity(x_hat))

        return out_morbidity, out_severity


class SerialMultiObjective(nn.Module):

    def __init__(self, cfg=None, backbone='resnet18', device=None, *args, **kwargs):
        super(SerialMultiObjective, self).__init__()
        self.config = cfg
        self.device = device

        self.cfg_morbidity = cfg['data']['modes']['morbidity']
        self.cfg_severity = cfg['data']['modes']['severity']

        self.classes_morbidity = self.cfg_morbidity['classes']
        self.classes_severity = self.cfg_severity['classes']

        self.labels = [*self.cfg_morbidity['classes'], *self.cfg_severity['classes']]
        self.class_to_id = {c: i for i, c in enumerate(self.labels)}
        # BACKBONE
        self.softmax = nn.Softmax(dim=1)
        self.backbone, self.in_features = get_backbone(backbone)
        self.drop_rate = self.config['model']['dropout_rate']

        # HEAD CLASSIFICATION : MORBIDITY
        print('HEAD CLASSIFICATION :', self.config['model']['head'])
        input_features = 24 if cfg['model']['structure_bx'] == 'brixia' else len(self.classes_severity)
        self.Head_Morbidity = nn.Sequential(
            OrderedDict([
                ('M_linear0', nn.Linear(in_features=input_features, out_features=32, bias=True)),
                ('M_ReLU0', nn.ReLU()),
                ('M_Dropout0', nn.Dropout(p=self.drop_rate)),
                ('M_linear1', nn.Linear(in_features=32, out_features=64, bias=True)),
                ('M_ReLU1', nn.ReLU()),
                ('M_Dropout1', nn.Dropout(p=self.drop_rate)),
                ('M_classification-Head', nn.Linear(in_features=64, out_features=len(self.classes_morbidity), bias=True))
            ]))
        init_weights(self.Head_Morbidity)
        if cfg['model']['structure_bx'] == 'brixia':
            self.global_ = nn.Sequential(OrderedDict(
                [('dropout_global', nn.Dropout(p=0.5, inplace=False)),
                 ('relu_global', nn.ReLU()),
                 ('linear_global', nn.Linear(in_features=self.in_features, out_features=128))]))
            self.area_A_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_A', nn.ReLU()),
                    ('area_A',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_B_output = nn.Sequential(OrderedDict(
                [('ReLU_B', nn.ReLU()),
                 ('area_B',
                  nn.Linear(in_features=128, out_features=4))]))
            self.area_C_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_C', nn.ReLU()),
                    ('area_C',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_D_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_D', nn.ReLU()),
                    ('area_D',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_E_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_E', nn.ReLU()),
                    ('area_E',
                     nn.Linear(in_features=128, out_features=4))]))
            self.area_F_output = nn.Sequential(OrderedDict(
                [
                    ('ReLU_F', nn.ReLU()),
                    ('area_F',
                     nn.Linear(in_features=128, out_features=4))]))

            for i, param in enumerate(list(self.backbone.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_A_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_B_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_C_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_D_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_E_output.parameters())):
                param.requires_grad = True
            for i, param in enumerate(list(self.area_F_output.parameters())):
                param.requires_grad = True
        else:
            self.Head_Severity = nn.Sequential(
                OrderedDict([
                    ('S_Dropout0', nn.Dropout(p=self.drop_rate)),
                    ('S_linear0', nn.Linear(in_features=self.in_features, out_features=128, bias=True)),
                    ('S_ReLU0', nn.ReLU()),
                    ('S_Dropout1', nn.Dropout(p=self.drop_rate)),
                    ('S_linear1', nn.Linear(in_features=128, out_features=32, bias=True)),
                    ('S_ReLU1', nn.ReLU()),
                    ('S_classification-Head', nn.Linear(in_features=32, out_features=len(self.classes_severity), bias=True))
                ]))

            init_weights(self.Head_Severity)

        # Turn on the training of the gradients
        change_head(model=self.backbone, model_name=backbone, new_head=nn.Identity())
        # Freeze Backbone
        if self.config['model']['freezing']:
            freeze_backbone(model=self, perc=0.70)

    def activate_Head_training_module(self):
        for i, param in enumerate(list(self.parameters())):
            param.requires_grad = True

    def load_backbone_average_weights(self, morbidity_params, severity_params, Beta=0.):
        """
        This function load the mean weights of the two models, and fuse them together with an adaptive mean. If Beta == 0, the weights
        loaded in the model are coming from only the severity model, if Beta == 1, the weights loaded in the model are coming from only the morbidity model.

        :param morbidity_params: Morbidity model parameters
        :param severity_params: Severity model parameters
        :param Beta: Beta parameter for the weighted mean
        :return: None
        """
        # TODO Adapt this one ?
        print('LOADING BACKBONE WEIGHTS (1/2 * S + 1/2 * M)')
        this_model_params = dict(self.named_parameters())
        for (name_m, param_morbidity), (name_s, param_severity) in zip(morbidity_params.items(), severity_params.items()):
            if 'module' in name_m:
                name_m = name_m.split('module.')[1]
            if 'module' in name_s:
                name_s = name_s.split('module.')[1]
            if name_m == name_s:

                if name_m in this_model_params and name_s in this_model_params:
                    this_model_params[name_m].data.copy_((Beta * param_morbidity.data + (1 - Beta) * param_severity.data))
        self.load_state_dict(this_model_params, strict=False)

    def forward(self, x):
        x_hat = self.backbone(x)
        if self.config['model']['structure_bx'] == 'brixia':
            # Areas and Global scorer
            out_global_memory = self.global_(x_hat)
            out_area_A = self.area_A_output(out_global_memory)[:, None, :]
            out_area_B = self.area_B_output(out_global_memory)[:, None, :]
            out_area_C = self.area_C_output(out_global_memory)[:, None, :]
            out_area_D = self.area_D_output(out_global_memory)[:, None, :]
            out_area_E = self.area_E_output(out_global_memory)[:, None, :]
            out_area_F = self.area_F_output(out_global_memory)[:, None, :]
            out_severity = torch.cat((out_area_A, out_area_B, out_area_C, out_area_D, out_area_E, out_area_F), dim=1)
            input_morbidity = torch.flatten(out_severity, start_dim=1)
            out_morbidity = self.softmax(self.Head_Morbidity(input_morbidity))
        else:
            out_severity = self.Head_Severity(x_hat)
            out_morbidity = self.softmax(self.Head_Morbidity(out_severity))
        return out_morbidity, out_severity
