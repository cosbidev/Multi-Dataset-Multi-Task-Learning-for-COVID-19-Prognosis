import __future__
import sys

import PIL
import numpy as np

print('Python %s on %s' % (sys.version, sys.platform))
import os
import argparse
import sys
from easydict import EasyDict as edict
import torch
import yaml
import pandas as pd
from tqdm import tqdm

from skimage import exposure, transform, measure
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from src import seed_all, mkdir, ImagesRawDataset, remove_small_regions, plot_evaluate_unet, loadData, save_mask

# Add the configuration
sys.argv.extend(
        [
            '--cfg_file', 'configs/img_unet_inference.yaml',

        ]
    )


# Configuration file
parser = argparse.ArgumentParser(description="Configuration File")
parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
args = parser.parse_args()




def inference_mask():
    import numpy as np
    # Configuration file opening
    with open(args.cfg_file) as file:
        cfg = edict(yaml.load(file, Loader=yaml.FullLoader))


    # Seed everything
    seed_all(cfg['seed'])


    # Parameters
    exp_name = cfg['exp_name']
    model_name = cfg['model']['model_name']

    print('### Configuration file: {}'.format(args.cfg_file))
    print('### Experiment: {}'.format(exp_name))
    print('### Model: {}'.format(model_name))


    # Device
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']

    # Files and Directories
    weights_directory = cfg.model.pretrained # folder where is stored the pretrained model


    # Directories where to save the masks
    masks_dir = cfg.data.masks.saving_dir
    mkdir(masks_dir)


    # DATASETS:
    # 1. imgs-r2 and imgs-r3:
    data_r2 = pd.read_excel(os.path.join(cfg.data.img_dir, 'AIforCOVID_r2.xlsx' ))
    data_r3 = pd.read_excel(os.path.join(cfg.data.img_dir, 'AIforCOVID_r3.xlsx' ))

    """
    dataset_r2 = ImagesRawDataset(data=data_r2, cfg_data=cfg.data.modes.masks, step='r2_dataset_inference', img_dir= )
    dataset_r3 = ImagesRawDataset(data=data_r3, cfg_data=cfg.data.modes.masks, step='r3_dataset_inference', img_dir= os.path.join(cfg.data.img_dir, cfg.data.dataset.r3))
    """

    # Categories
    categories = ["MILD", "SEVERE"]
    y_label = "Prognosis"

    # Model path
    model_path = 'models/segmentation_brixia/trained_model.hdf5'
    im_shape = (256, 256)

    # Clinical data
    meta_path_2 = os.path.join(cfg.data.img_dir, 'AIforCOVID_r2.xlsx')
    meta_path_3 = os.path.join(cfg.data.img_dir, 'AIforCOVID_r3.xlsx')

    # Load clinical data and concatenate
    clinical_meta_2 = pd.read_excel(meta_path_2)
    clinical_meta_3 = pd.read_excel(meta_path_3)
    clinical_meta_global = pd.concat([clinical_meta_2, clinical_meta_3]).set_index('ImageFile')

    # Bounding-box file
    bounding_box_file = os.path.join("data/AIforCovid/processed", "box_data.xlsx")

    # Data directories
    # Load test data
    img_dir_r2 = cfg.data.dataset.r2
    img_dir_r3 = cfg.data.dataset.r3

    # Load data
    X, original_shape, imgs_name = loadData(im_shape, [img_dir_r2, img_dir_r3])

    # Load U-net
    UNet = load_model(model_path)


    n_test = X.shape[0]
    inp_shape = X[0].shape


    # For inference standard keras ImageGenerator is used.
    test_gen = ImageDataGenerator(rescale=1.)

    i = 0
    # Bounding Boxes
    dx_box = []
    sx_box = []
    all_box = []
    y_s = []
    imgs_id = []


    # MASK DIRECTORY:
    # Create the directory where to save the masks
    masks_dir = cfg.data.masks.saving_dir


    for xx, img_name in tqdm(zip(X, imgs_name)):
        # Get image name

        pred = UNet.predict(np.expand_dims(xx, axis=0))[..., 0].reshape(inp_shape[:2])

        clinical_data = clinical_meta_global.loc[img_name.split('/')[-1].split('.')[0]]
        id_patient = img_name.split('/')[-1].split('.')[0]
        y_patient = clinical_data[y_label]
        y_s.append(y_patient)
        imgs_id.append(id_patient)
        # Binarize masks
        pr = pred > 0.5

        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

        # resize to original shape
        pr = transform.resize(pr, original_shape[i])

        # Produce a mask AND image
        if 'P_2_56' in img_name:

            or_ = transform.resize(xx, original_shape[i])
            pr_or_and = or_[:, :, 0] * pr


        # Save the mask
        save_mask(pr, masks_dir, img_name.split('/')[-1].split('.')[0] + '.tif')

        # get box for single lungs
        lbl = measure.label(pr)

        props = measure.regionprops(lbl)

        if len(props) >= 2:
            box_1 = props[0].bbox
            box_2 = props[1].bbox
            if box_1[1] < box_2[1]:
                dx_box.append(list(box_1))
                sx_box.append(list(box_2))
            else:
                dx_box.append(list(box_2))
                sx_box.append(list(box_1))
            # get box for both lungs
            props = measure.regionprops(pr.astype("int64"))
            if len(props) == 1:
                all_box.append(list(props[0].bbox))
            else:
                all_box.append([0, 0, lbl.shape[0], lbl.shape[1]])
        else:
            dx_box.append(None)
            sx_box.append(None)
            all_box.append([0, 0, lbl.shape[0], lbl.shape[1]])

        i += 1
        if i == n_test:
            break

    # save excel with boxes
    bounding_box = pd.DataFrame(index=imgs_id)
    bounding_box["dx"] = dx_box
    bounding_box["sx"] = sx_box
    bounding_box["all"] = all_box
    bounding_box["label"] = y_s
    bounding_box["img_path"] = imgs_name

    # dropna
    bounding_box = bounding_box.dropna(subset=["label"])

    bounding_box.to_excel(bounding_box_file, index=True, index_label="img")


if __name__ == '__main__':
    inference_mask()