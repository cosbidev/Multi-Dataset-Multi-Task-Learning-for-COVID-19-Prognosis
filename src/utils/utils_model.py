import itertools
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
    if  "resnet" in model_name or "shufflenet" in model_name or "resnext" in model_name:
        model.fc = new_head
    elif "mnasnet" in model_name or "mobilenet" in model_name or "alexnet" in model_name:
        model.classifier[-1] = new_head
    elif "densenet" in model_name:
        model.classifier = new_head

def get_backbone(model_name= ''):

    # Finetuning the convnet
    print("********************************************")
    if model_name == "resnet18":
        model = models.resnet18(weights=True)
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
        """    elif model == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=True)
        in_features = model.classifier[1].in_features 
    elif model == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1), stride=(1, 1))
        """

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
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=True)
        in_features = model.fc.in_features
        # model.fc = nn.Linear(2048, len(class_names), bias=True)
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
        in_features = model.fc.in_features
        # model.fc = nn.Linear(2048, len(class_names), bias=True)
    elif model_name == "mnasnet0_5":
        model = models.mnasnet0_5(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
        """ 
    elif model_name == "vgg11":
        model = models.vgg11(pretrained=True)
        in_features = model.classifier[0].in_features
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
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
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
    elif model_name == "vgg19_bn":
        model = models.vgg19_bn(pretrained=True)
        in_features = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(class_names), bias=True)
        """
    return model, in_features



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

def train_severity(model,
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
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(total=len(dataloaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
                for inputs, labels, file_name in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.float())
                        _, preds = torch.max(outputs, dim=1)


                        labels = labels.type(torch.float32)
                        labels = labels.unsqueeze(1)
                        loss = criterion(outputs, labels)
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(inputs.shape[0])



            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == 'val':
                scheduler.step(epoch_loss)
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
            if phase == 'val':
                if epoch_acc > best_acc:
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

        print()

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(model, os.path.join(model_dir, model_file_name))

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
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(total=len(dataloaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
                for inputs, labels, file_name in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # Train updating of weights by gradient descent

                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(inputs.float())

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
                scheduler.step(epoch_loss)
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
            if phase == 'val':
                if epoch_acc > best_acc and epoch!=0:
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
    torch.save(model, os.path.join(model_dir, model_file_name))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def accuracy(output, target, topk=(1, )):
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

def get_Model(kind='parallel', config_model=None, device='', **kwargs):
    """
    Thid function returns the MultiObjective Model
    Args:
        kind:
        config_model:
        device:

    Returns:

    """

    if kind == 'parallel':
        return ParallelMultiObjective(device=device, config_model=config_model, **kwargs)
    elif kind == 'serial':
        
        # TODO Serial Class
        
        pass
    elif config_model['taks'] == 'morbidity':
        return MorbidityModel(device=device, config_model=config_model, **kwargs)

class MorbidityModel(nn.Module):

    def __init__(self, cfg=None, backbone='resnet18', device=None, freezing_layer = '', *args, **kwargs):



        super(MorbidityModel, self).__init__()
        layers_backbone = {'last': 2, 'last-1': 3}
        self.config = cfg
        self.device = device
        self.classes = cfg['data']['classes']
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        # BACKBONE
        self.backbone, in_features = get_backbone(backbone)

        # HEAD CLASSIFICATION
        New_classification_Head = nn.Sequential(OrderedDict(
             [  ('classification-Head', nn.Linear(in_features=in_features, out_features=len(self.classes), bias=True))]))
        # CHANGE HEAD
        for i, param in enumerate(list(New_classification_Head.parameters())):
            param.requires_grad = True

        change_head(model=self.backbone, model_name=backbone, new_head=New_classification_Head)

        # Freeze Backbone
        """        for i, param in enumerate(list(self.backbone.parameters())[::-1]):
            if i == 0 or i == 1:
                continue
            if freezing_layer in layers_backbone.keys():
                if i == layers_backbone[freezing_layer]:
                    continue

            param.requires_grad = False
        """


    def forward(self, x):
        x = self.backbone(x)
        return x



        # (AIFORCOVID) MORBIDITY HEAD
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            pass


class SeverityModel(nn.Module):

    def __init__(self, cfg=None, backbone='resnet18', device=None, freezing_layer = '', *args, **kwargs):



        super(SeverityModel, self).__init__()
        layers_backbone = {'last': 2, 'last-1': 3}
        self.config = cfg
        self.device = device
        self.labels = cfg['data']['classes']
        self.class_to_id = {c: i for i, c in enumerate(self.labels)}
        # BACKBONE
        self.backbone, in_features = get_backbone(backbone)
        # HEAD CLASSIFICATION
        if 'vgg' in backbone:
            New_classification_Head = nn.Sequential(OrderedDict(
                [('hidden1-Head', nn.Linear(in_features=in_features, out_features=4096, bias=True)),
                 ('relu1-Head', nn.ReLU()),
                 ('dropout1-Head', nn.Dropout(p=cfg['model']['dropout_rate'])),
                 ('hidden2-Head', nn.Linear(in_features=4096, out_features=4096 * 2, bias=True)),
                 ('relu2-Head', nn.ReLU()),
                 ('dropout2-Head', nn.Dropout(p=cfg['model']['dropout_rate'])),
                 ('hidden3-Head', nn.Linear(in_features=4096 * 2, out_features=2048, bias=True)),
                 ('relu3-Head', nn.ReLU()),
                 ('dropout3-Head', nn.Dropout(p=cfg['model']['dropout_rate'])),
                 ('classification-Head', nn.Linear(in_features=in_features, out_features=6, bias=True))]))
        else:
            New_classification_Head = nn.Sequential(OrderedDict(
                 [  ('dropout0-Head', nn.Dropout(p=cfg['model']['dropout_rate'])),
                    ('hidden1-Head', nn.Linear(in_features=in_features, out_features=2 * in_features, bias=True)),
                    ('relu1-Head', nn.ReLU()),
                    ('dropout1-Head', nn.Dropout(p=cfg['model']['dropout_rate'])),
                    ('hidden2-Head', nn.Linear(in_features=2 * in_features, out_features=in_features, bias=True)),
                    ('relu2-Head', nn.ReLU()),
                    ('dropout2-Head', nn.Dropout(p=cfg['model']['dropout_rate'])),
                    ('classification-Head', nn.Linear(in_features=in_features, out_features=6, bias=True))]))
        # CHANGE HEAD
        for i, param in enumerate(list(New_classification_Head.parameters())):
            param.requires_grad = True

        change_head(model=self.backbone, model_name=backbone, new_head=New_classification_Head)

        # Freeze Backbone
        """        for i, param in enumerate(list(self.backbone.parameters())[::-1]):
            if i == 0 or i == 1:
                continue
            if freezing_layer in layers_backbone.keys():
                if i == layers_backbone[freezing_layer]:
                    continue

            param.requires_grad = False
        """


    def forward(self, x):
        x = self.backbone(x)
        return x



        # (AIFORCOVID) MORBIDITY HEAD
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            pass

def get_SingleTaskModel(kind='morbidity', backbone = '', cfg=None, device=None, *args, **kwargs ):
    if cfg['model']['task'] == 'morbidity':
        return MorbidityModel(cfg=cfg, backbone=backbone,device=device, *args, **kwargs)
    if cfg['model']['task'] == 'severity':
        return SeverityModel(cfg=cfg, backbone=backbone,device=device, *args, **kwargs)








class ParallelMultiObjective(nn.Module):
    def __init__(self, device, config_model, backbone='resnet18',  dropout_rate=0.5, ):
        super(ParallelMultiObjective, self).__init__()
        self.config_model = config_model
        self.device = device

        # BACKBONE
        self.backbone = get_backbone(backbone)

        # (AIFORCOVID) MORBIDITY HEAD

        self.linear_h1 = nn.Sequential(
            nn.Linear(dim1_og, dim1), nn.ReLU())

        ### Model 2
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim1_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim2_og + dim1_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Model 3
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec3) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec3), dim=1))  # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_h1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec2, vec1) if self.use_bilinear else self.linear_z2(torch.cat((vec2, vec1), dim=1))  # Gate Graph with Omic
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_h2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(torch.cat((vec1, vec3), dim=1))  # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_h3(vec3)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1, device=self.device).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1, device=self.device).fill_(1)), 1)
        o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1, device=self.device).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out







