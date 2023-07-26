import matplotlib.pyplot as plt
import time
import copy
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torch

from src.utils.util_components import *


def get_unet(model_name):
    if model_name == "guided":
        return GuidedUNet()
    else:
        raise ValueError(model_name)


class GuidedDecoder_1(nn.Module):
    def __init__(self):
        super(GuidedDecoder_1, self).__init__()
        self.guided_decoder = nn.Sequential(
            # N, 512, 4, 4
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.guided_decoder(x)


class GuidedDecoder_2(nn.Module):
    def __init__(self):
        super(GuidedDecoder_2, self).__init__()
        self.guided_decoder = nn.Sequential(
            # N, 256, 8, 8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.guided_decoder(x)


class GuidedDecoder_3(nn.Module):
    def __init__(self):
        super(GuidedDecoder_3, self).__init__()
        self.guided_decoder = nn.Sequential(
            # N, 128, 16, 16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.guided_decoder(x)


class GuidedDecoder_4(nn.Module):
    def __init__(self):
        super(GuidedDecoder_4, self).__init__()
        self.guided_decoder = nn.Sequential(
            # N, 64, 32, 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.guided_decoder(x)


class GuidedDecoder_5(nn.Module):
    def __init__(self):
        super(GuidedDecoder_5, self).__init__()
        self.guided_decoder = nn.Sequential(
            # N, 32, 64, 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.guided_decoder(x)


class GuidedDecoder_6(nn.Module):
    def __init__(self):
        super(GuidedDecoder_6, self).__init__()
        self.guided_decoder = nn.Sequential(
            # N, 16, 128, 128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.guided_decoder(x)


class GuidedUNet(nn.Module):
    def __init__(self, bilinear=True):
        super(GuidedUNet, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(1, 16)
        self.down1 = DownConv(16, 32)
        self.down2 = DownConv(32, 64)
        self.down3 = DownConv(64, 128)
        self.down4 = DownConv(128, 256)
        self.down5 = DownConv(256, 512)
        factor = 2 if bilinear else 1
        self.down6 = DownConv(512, 1024 // factor)

        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 1024)  # N, 1024
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(1024, 512*4*4),  # N, 1024
            nn.Unflatten(1, (512, 4, 4)),  # N, 512, 4, 4
        )

        self.up1 = UpConv(1024, 512 // factor)
        self.up2 = UpConv(512, 256 // factor)
        self.up3 = UpConv(256, 128 // factor)
        self.up4 = UpConv(128, 64 // factor)
        self.up5 = UpConv(64, 32 // factor)
        self.up6 = UpConv(32, 16)
        self.outc = OutConv(16, 1)

        #self.decoder_guide_pre = GuidedDecoder_1()
        self.decoder_guide_1 = GuidedDecoder_1()
        self.decoder_guide_2 = GuidedDecoder_2()
        self.decoder_guide_3 = GuidedDecoder_3()
        self.decoder_guide_4 = GuidedDecoder_4()
        self.decoder_guide_5 = GuidedDecoder_5()
        self.decoder_guide_6 = GuidedDecoder_6()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x8 = self.fc_1(x7) # h
        x9 = self.fc_2(x8)

        x10 = self.up1(x9, x6)
        x11 = self.up2(x10, x5)
        x12 = self.up3(x11, x4)
        x13 = self.up4(x12, x3)
        x14 = self.up5(x13, x2)
        x15 = self.up6(x14, x1)
        x16 = self.outc(x15)

        #decode_pre = self.decoder_guide_pre(x7)
        decode_1 = self.decoder_guide_1(x9)
        decode_2 = self.decoder_guide_2(x10)
        decode_3 = self.decoder_guide_3(x11)
        decode_4 = self.decoder_guide_4(x12)
        decode_5 = self.decoder_guide_5(x13)
        decode_6 = self.decoder_guide_6(x14)

        return decode_1, decode_2, decode_3, decode_4, decode_5, decode_6, x16


def train_unet(model, data_loaders, criterion, optimizer, scheduler, num_epochs, early_stopping, model_dir,
               device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.Inf

    history = {'train_loss': [], 'val_loss': []}

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

            # Iterate over data.
            with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
                for input, labels, id in data_loaders[phase]:
                    input = input.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs_1, outputs_2, outputs_3, outputs_4, outputs_5, outputs_6, outputs_decoder = model(input.float())  # outputs_decoder -- output U-NET

                        loss_1 = criterion(outputs_1.float(), input.float())
                        loss_2 = criterion(outputs_2.float(), input.float())
                        loss_3 = criterion(outputs_3.float(), input.float())
                        loss_4 = criterion(outputs_4.float(), input.float())
                        loss_5 = criterion(outputs_5.float(), input.float())
                        loss_6 = criterion(outputs_6.float(), input.float())
                        loss_decoder = criterion(outputs_decoder.float(), input.float())
                        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_decoder
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * input.size(0)

                    pbar.update(input.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)

            if phase == 'val':
                scheduler.step(epoch_loss)

            # update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= early_stopping:
                        print(f'\nEarly Stopping! Total epochs: {epoch}%')
                        early_stop = True
                        break

        if early_stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return model, history


def evaluate_unet(model, data_loader, criterion, device):
    # Global and Class Metric
    metric_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}
    total_pred = {classname: 0 for classname in list(data_loader.dataset.idx_to_class.values()) + ["all"]}

    # Test loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):  # per batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            outputs_1, outputs_2, outputs_3, outputs_4, outputs_5, outputs_6, outputs_decoder = model(inputs.float())
            loss = criterion(outputs_decoder.float(), inputs.float())
            # global
            metric_pred['all'] += loss.item()
            total_pred['all'] += 1
            # class
            for label, input, output in zip(labels, inputs, outputs_decoder):
                metric_pred[data_loader.dataset.idx_to_class[label.item()]] += criterion(output.float(), input.float()).item()
                total_pred[data_loader.dataset.idx_to_class[label.item()]] += 1

    # Metric Mean
    test_results = {k: metric_pred[k] / total_pred[k] for k in metric_pred.keys() & total_pred}

    return test_results


def plot_evaluate_unet(model, data_loader, plot_dir, device):
    # Test loop
    model.eval()
    with torch.no_grad():
        for inputs, labels, file_names in tqdm(data_loader):  # per batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Prediction
            outputs_1, outputs_2, outputs_3, outputs_4, outputs_5, outputs_6, outputs_decoder = model(inputs.float())
            # Plot
            for patient_id, input, output in zip(file_names, inputs, outputs_decoder):
                input = input.cpu().detach().numpy()[0]
                output = output.cpu().detach().numpy()[0]

                plt.figure()
                plt.gray()
                plt.subplot(1, 2, 1)
                plt.imshow(input)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(output)
                plt.axis('off')
                plt.savefig(os.path.join(plot_dir, patient_id), dpi=300)
                plt.show()
