#!/bin/bash

python3 ./src/model/train_img_autoencoder.py -f ./configs/10/img/img_autoencoder_conv_1024.yaml -g cuda:1
python3 ./src/model/train_img_autoencoder.py -f ./configs/10/img/img_autoencoder_conv_2048.yaml -g cuda:1
python3 ./src/model/train_img_autoencoder.py -f ./configs/10/img/img_autoencoder_conv_4096.yaml -g cuda:1
python3 ./src/model/train_img_autoencoder.py -f ./configs/10/img/img_autoencoder_conv_8192.yaml -g cuda:1
python3 ./src/model/train_img_autoencoder.py -f ./configs/10/img/img_autoencoder_conv_16384.yaml -g cuda:1
