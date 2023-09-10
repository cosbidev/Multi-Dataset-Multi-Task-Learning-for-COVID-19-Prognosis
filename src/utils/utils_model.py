import itertools
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import pandas as pd
from scipy.stats import pearsonr
from skimage.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, r2_score, \
    explained_variance_score
from tqdm import tqdm
import os
from skimage import color
import math
from torchvision import models

model_list = ["alexnet",
              "vgg11",
              "vgg11_bn",
              "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
              "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "squeezenet1_0", "squeezenet1_1",
              "densenet121", "densenet169", "densenet161", "densenet201", "googlenet", "shufflenet_v2_x0_5",
              "shufflenet_v2_x1_0", "mobilenet_v2", "resnext50_32x4d", "wide_resnet50_2", "mnasnet0_5", "mnasnet1_0"]


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
    elif "vgg" in model_name or "mnasnet" in model_name or "mobilenet" in model_name or "alexnet" in model_name:
        model.classifier[-1] = new_head
    elif "densenet" in model_name:
        model.classifier = new_head
    elif "squeezenet" in model_name:
        model.classifier[1] = new_head


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
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
        in_features = model.fc.in_features
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        # model.classifier = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
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


def evaluate(model, test_loader, criterion, idx_to_class, device, topk=(1, 5)):
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

    model.eval()
    with torch.no_grad():

        # Testing loop
        for data, targets, file_name in tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            # Raw model output
            out = model(data.float())
            # Iterate through each example
            for pred, true in zip(out, targets):
                # Find topk accuracy
                acc_results[i, :] = accuracy(pred.unsqueeze(0), true.unsqueeze(0), topk)
                classes.append(idx_to_class[true.item()])
                # Calculate the loss
                loss = criterion(pred.view(1, len(idx_to_class)), true.view(1))
                losses.append(loss.item())
                i += 1

    # Send results to a dataframe and calculate average across classes
    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    results['class'] = classes
    results['loss'] = losses
    results = results.groupby(classes).mean()
    acc = acc_results.mean()

    return results.reset_index().rename(columns={'index': 'class'}), acc


def get_metrics_regression(y_true, y_pred):
    """
    This function computes the metrics for the regression task, MSE, L1, R2, explained variance, Correlation coefficient,
    and std deviation between predictions and true values
    """

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

    # REGIONS METRICS:
    y_true_region_R = np.sum(y_true[:, :3], axis=1)
    y_true_region_L = np.sum(y_true[:, 3:6], axis=1)
    y_pred_region_R = np.sum(y_pred[:, :3], axis=1)
    y_pred_region_L = np.sum(y_pred[:, 3:6], axis=1)

    corr_RL, _ = pearsonr(y_true_region_R, y_pred_region_R)
    corr_LL, _ = pearsonr(y_true_region_L, y_pred_region_L)
    std_RL = np.std(np.abs(y_true_region_R - y_pred_region_R))
    std_LL = np.std(np.abs(y_true_region_L - y_pred_region_L))

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

    return results.T


def train_severity(model,

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
    history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}

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
                for k, (inputs, labels, file_name) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())

                        labels = labels.type(torch.float32)

                        # This criterion calculate MSE over single areas: (MEAN-SQUARED ERROR)
                        if regression_type == 'area':
                            labels_single_areas = labels[:, :6]
                            loss = criterion(outputs, labels_single_areas)
                        elif regression_type == 'global':
                            # This criterion calculate MSE over global score: (MEAN-SQUARED ERROR)
                            labels_global = labels[:, 9]
                            loss = criterion(outputs, labels_global)
                        elif regression_type == 'region':
                            # This criterion calculate MSE over single regions: (MEAN-SQUARED ERROR)
                            labels_regions = labels[:, 6:9]
                            loss = criterion(outputs, labels_regions)

                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        # Collect predictions and true labels

                        if k == 0:
                            output_predictions = outputs.cpu().detach().numpy()
                            real_target = labels.cpu().detach().numpy()
                        else:
                            output_predictions = np.concatenate((output_predictions, outputs.cpu().detach().numpy()), 0)
                            real_target = np.concatenate((real_target, labels.cpu().detach().numpy()), 0)
                        # real_target = np.append(real_target, labels.cpu().detach().numpy())
                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    pbar.update(inputs.shape[0])

            metrics_epoch = get_metrics_regression(real_target[:, :6], output_predictions)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_metrics'].append(metrics_epoch)

            else:
                history['val_loss'].append(epoch_loss)
                history['val_metrics'].append(metrics_epoch)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if epoch > model.config['trainer']['warmup_epochs']:
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
    torch.save(model, os.path.join(model_dir, model_file_name))

    return model, history


def train_MultiTask(model,
                    optimizer,
                    scheduler,
                    criterion,
                    model_file_name,
                    dataloaders,
                    model_dir,
                    device,
                    num_epochs=25,
                    max_epochs_stop=3,
                    regression_type='area'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_accuracy = 0.0

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
                for inputs, labels, file_name, dataset_class in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # Train updating of weights by gradient descent

                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(inputs.float())
                        # Outputs of the two heads
                        Morbidity_head_outputs = outputs[0]
                        Severity_head_outputs = outputs[1]

                        labels = labels.type(torch.float32)
                        if model.config['model']['softmax']:
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
            epoch_loss_AFC = running_loss['Loss_M'] / count_AFC
            epoch_loss_BX = running_loss['Loss_S'] / count_BX

            if phase == 'val':
                val_loss = epoch_loss_tot
                update_learning_rate(optimizer=optimizer, scheduler=scheduler, metric=val_loss)
            epoch_acc = running_corrects.double() / count_AFC

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
            if epoch > model.config['trainer']['warmup_epochs']:
                if phase == 'val':
                    if epoch_loss_tot < best_loss:
                        best_epoch = epoch
                        best_epoch_accuracy = epoch_acc
                        best_loss = epoch_loss_tot
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

    # Save model
    torch.save(model, os.path.join(model_dir, model_file_name + '_{0}_.pt'.format(best_epoch)))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history


def train_morbidity(model,
                    criterion,
                    optimizer,
                    scheduler,
                    model_file_name,
                    dataloaders,
                    model_dir,
                    device,
                    num_epochs=25,
                    max_epochs_stop=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                for inputs, labels, file_name in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # Train updating of weights by gradient descent

                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(inputs.float())
                        if model.config['model']['softmax']:
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
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch > model.config['trainer']['warmup_epochs']:
                if phase == 'val':
                    if epoch_acc > best_acc and epoch != 0:
                        best_epoch = epoch
                        best_acc = epoch_acc
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

    # Save model
    torch.save(model, os.path.join(model_dir, model_file_name + '_{0}_.pt'.format(best_epoch)))

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
def evaluate_multi_task(model, test_loader, criterion, idx_to_class, device, topk=(1, 5)):
    """Measure the performance of a trained PyTorch model"""
    pass

def evaluate_regression(model, test_loader, criterion, idx_to_class, device, regression_type='area'):
    """Measure the performance of a trained PyTorch model on a regression task
    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
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

        # Testing loop

        # Iterate over data.
        number_samples = 0
        for k, (inputs, labels, file_name) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            # track history if only in train
            outputs = model(inputs.float())
            labels = labels.type(torch.float32)
            # This criterion calculate MSE over single areas: (MEAN-SQUARED ERROR)
            if regression_type == 'area':
                labels_single_areas = labels[:, :6]
                loss = criterion(outputs, labels_single_areas)
            elif regression_type == 'global':
                # This criterion calculate MSE over global score: (MEAN-SQUARED ERROR)
                labels_global = labels[:, 9]
                loss = criterion(outputs, labels_global)
            elif regression_type == 'region':
                # This criterion calculate MSE over single regions: (MEAN-SQUARED ERROR)
                labels_regions = labels[:, 6:9]
                loss = criterion(outputs, labels_regions)
            losses += loss.item() * inputs.size(0)
            if k == 0:
                output_predictions = outputs.cpu().detach().numpy()
                real_target = labels.cpu().detach().numpy()
            else:
                output_predictions = np.concatenate((output_predictions, outputs.cpu().detach().numpy()), 0)
                real_target = np.concatenate((real_target, labels.cpu().detach().numpy()), 0)

    loss_test = losses / len(test_loader.dataset)
    metrics_test = get_metrics_regression(real_target[:, :6], output_predictions)
    return metrics_test, loss_test


def evaluate(model, test_loader, criterion, idx_to_class, device, topk=(1, 5)):
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

        # Testing loop
        for data, targets, file_name in tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            # Raw model output
            outputs = model(data.float())
            if model.config['model']['softmax']:
                outputs = nn.Softmax(dim=1)(outputs)

            _, preds = torch.max(outputs, 1)
            _, labels_gt = torch.max(targets, 1)

            true_labels.append(labels_gt.data.cpu().numpy())
            predicted_labels.append(preds.data.cpu().numpy())

            # Iterate through each example

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
    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    results['class'] = classes
    results['loss'] = losses

    results = results.groupby(classes).mean()
    results['class'] = results['class'].apply(lambda x: idx_to_class[x])
    acc = acc_results.mean()
    return results, {
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

        # HEAD CLASSIFICATION
        if 'squeezenet' in backbone:
            New_classification_Head = nn.Sequential(OrderedDict(
                [('classification-Head',
                  nn.Conv2d(512, len(self.classes), kernel_size=(1, 1), stride=(1, 1))
                  )]
            ))
        else:
            New_classification_Head = nn.Sequential(OrderedDict(
                [('classification-Head',
                  nn.Linear(in_features=in_features, out_features=len(self.classes), bias=True))]))
        # CHANGE HEAD
        for i, param in enumerate(list(New_classification_Head.parameters())):
            param.requires_grad = True

        # (AIFORCOVID) MORBIDITY HEAD
        change_head(model=self.backbone, model_name=backbone, new_head=New_classification_Head)

        if self.config['model']['freezing']:
            start_counter_seq = 0
            # Freeze Backbone
            layers_seq_to_freeze = np.ceil(
                len([module for module in list(self.backbone.modules())[1:] if isinstance(module, nn.Sequential)]) / 5)
            for module in list(self.backbone.modules())[1:]:
                print('ANALIZE THIS: ', module.__class__)
                if isinstance(module, nn.Sequential):
                    start_counter_seq += 1
                    print('***** FREEZING *****: SEQ layer: ', module.__class__)
                    for i, param in enumerate(list(module.parameters())):
                        param.requires_grad = False

                print('***** FREEZING *****: SING layer: ', module.__class__)
                for i, param in enumerate(list(module.parameters())):
                    param.requires_grad = False

                if start_counter_seq == layers_seq_to_freeze:
                    break

    def forward(self, x):
        x = self.backbone(x)
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

        self.softmax = nn.Softmax(dim=1)
        self.backbone, in_features = get_backbone(backbone)
        # HEAD CLASSIFICATION
        if 'squeezenet' in backbone:
            New_classification_Head = nn.Sequential(OrderedDict(
                [('classification-Head',
                  nn.Conv2d(512, len(self.labels), kernel_size=(1, 1), stride=(1, 1))
                  )]
            ))
        else:
            New_classification_Head = nn.Sequential(OrderedDict(
                [(
                    'classification-Head', nn.Linear(in_features=in_features, out_features=len(self.labels), bias=True))]))
        # CHANGE HEAD
        for i, param in enumerate(list(New_classification_Head.parameters())):
            param.requires_grad = True

        change_head(model=self.backbone, model_name=backbone, new_head=New_classification_Head)

        start_counter_seq = 0
        # Freeze Backbone
        layers_seq_to_freeze = np.ceil(
            len([module for module in list(self.backbone.modules())[1:] if isinstance(module, nn.Sequential)]) / 5)
        for module in list(self.backbone.modules())[1:]:
            print('ANALIZE THIS: ', module.__class__)
            if isinstance(module, nn.Sequential):
                start_counter_seq += 1
                print('***** FREEZING *****: SEQ layer: ', module.__class__)
                for i, param in enumerate(list(module.parameters())):
                    param.requires_grad = False

            print('***** FREEZING *****: SING layer: ', module.__class__)
            for i, param in enumerate(list(module.parameters())):
                param.requires_grad = False

            if start_counter_seq == layers_seq_to_freeze:
                break

        change_head(model=self.backbone, model_name=backbone, new_head=New_classification_Head)

    def forward(self, x):
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
        return SeverityModel(cfg=cfg, backbone=backbone, device=device, *args, **kwargs)


def MSE_loss(outputs, labels):

    number =  (len(outputs) - len(labels[torch.all(torch.isnan(outputs), dim=1)])) if (len(outputs) - len(labels[torch.all(torch.isnan(outputs), dim=1)])) != 0 else 1

    return torch.nansum((outputs - labels) ** 2) / number


class IdentityMultiHeadLoss(torch.nn.Module):
    def __init__(self, cfg: dict, eps: float = 1e-08):
        super(IdentityMultiHeadLoss, self).__init__()
        self.cfg = cfg
        self.regression = self.cfg.model.regression_type
        self.loss_1 = cfg.trainer.loss_1
        self.loss_2 = cfg.trainer.loss_2
        self.encode_dataset = {'AFC': 0, 'BX': 1}
        self.dim_1 = 2
        encode_bx_dim = {'area': 6, 'region': 3, 'global': 1}
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
            return MSE_loss
        elif name.upper() == 'BCE':
            return nn.BCELoss()
        elif name.upper() == 'CE':
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def forward(self, outputs, labels, **kwargs):

        labels = labels[:, :6]
        labels_class = kwargs['dataset_class']
        # Create a new vector with the labels that link each sample to the corresponding dataset
        labels_class = torch.Tensor(self.encode_batch(labels_class)).to(labels.device)
        BX_mask = self.get_brixia_mask(labels=labels_class, dim=self.dim_2).to(labels.device)
        AFC_mask = self.get_aiforcovid_mask(labels=labels_class, dim=self.dim_1).to(labels.device)

        BX_selector = ~torch.any(BX_mask == 0, dim=1)
        AFC_selector = ~torch.any(AFC_mask == 0, dim=1)

        # Calculate the loss function for each head/Task
        Loss_AFC = self.get_loss(self.loss_1)
        Loss_BX = self.get_loss(self.loss_2)

        # Calculate the loss for each head
        loss_1 = Loss_AFC(outputs[0] * AFC_mask, labels[:, :self.dim_1] * AFC_mask)
        loss_2 = Loss_BX(outputs[1] * BX_mask, labels[:, :self.dim_2] * BX_mask)

        # Calculate the total loss
        Loss_TOT = loss_1 + loss_2
        return {'Loss_TOT': Loss_TOT, 'Loss_M': loss_1, 'Loss_S': loss_2}, {'AFC_sel': AFC_selector, 'BX_sel': BX_selector}

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


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
        self.train_backbone = True


        # HEAD CLASSIFICATION : MORBIDITY
        # HEAD CLASSIFICATION : MORBIDITY
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
        init_weights(self.Head_Morbidity)
        init_weights(self.Head_Severity)

        # Turn on the training of the gradients
        change_head(model=self.backbone, model_name=backbone, new_head=nn.Identity())

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
        this_model_params = dict(self.named_parameters())
        for (name_m, param_morbidity), (name_s, param_severity) in zip(morbidity_params, severity_params):
            if name_m == name_s:
                print('ANALYZING: (S) ', name_s, ' ; (M) ', name_m)
                if name_m in this_model_params and name_s in this_model_params:
                    this_model_params[name_m].data.copy_((Beta * param_morbidity.data + (1 - Beta) * param_severity.data))
        self.load_state_dict(this_model_params, strict=False)

    def forward(self, x):
        x_hat = self.backbone(x)
        out1 = self.Head_Morbidity(x_hat)
        out2 = self.Head_Severity(x_hat)
        return out1, out2


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

        # Turn on the training of the gradients
        change_head(model=self.backbone, model_name=backbone, new_head=nn.Identity())

    def activate_Head_training_module(self):
        for i, param in enumerate(list(self.parameters())):
            param.requires_grad = True

    def load_mean_weights(self, morbidity_params, severity_params, Beta=0.):
        """
        This function load the mean weights of the two models, and fuse them together with an adaptive mean. If Beta == 0, the weights
        loaded in the model are coming from only the severity model, if Beta == 1, the weights loaded in the model are coming from only the morbidity model.

        :param morbidity_params: Morbidity model parameters
        :param severity_params: Severity model parameters
        :param Beta: Beta parameter for the weighted mean
        :return: None
        """
        # TODO Adapt this one ?
        this_model_params = dict(self.named_parameters())
        for (name_m, param_morbidity), (name_s, param_severity) in zip(morbidity_params, severity_params):
            if name_m == name_s:
                print('ANALYZING: (S) ', name_s, ' ; (M) ', name_m)
                if name_m in this_model_params and name_s in this_model_params:
                    this_model_params[name_m].data.copy_((Beta * param_morbidity.data + (1 - Beta) * param_severity.data))
        self.load_state_dict(this_model_params, strict=False)

    def forward(self, x):
        x_hat = self.backbone(x)
        out1 = self.Head_Morbidity(x_hat)
        out2 = self.Head_Severity(x_hat)
        return out1, out2
