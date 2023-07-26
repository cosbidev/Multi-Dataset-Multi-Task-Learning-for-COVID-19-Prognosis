import numpy as np
import pandas as pd
import os
from PIL import Image

# Opening the data dor R2-R3
data_box_r23 = pd.read_excel('data/AIforCovid/processed/box_data.xlsx')

# Opening the data for R1
data_box_r1 = pd.read_excel('/Users/filruff/Desktop/PHD/PROGETTI/ITA-CINA/ANNO1/CODICE_ITA_CHINA/ItaChinaCovid19/AIforCOVID2/data/processed/data.xlsx')
data_box_r1['img'] = [value.replace('.dcm','') for value in data_box_r1['img']]
# find the mask inside the directories
data_box_r1['img_path'] = np.nan

# Find masks in R1
masks_in_dir = [value.replace('.tif','') for value in os.listdir('data/AIforCOVID/masks')]



for mask_ in masks_in_dir:
    # open mask in R1
    print('processing: ', mask_)
    mask = Image.open('data/AIforCOVID/masks/' + mask_ + '.tif')
    mask = np.array(mask, dtype=np.uint8) * 255
    # Save in the new directory:
    path_dicom = 'data/AIforCOVID/imgs/' + mask_ + '.dcm'

    Image.fromarray(mask).save('data/AIforCOVID/processed/masks/' + mask_ + '.tif')


    if mask_ not in data_box_r1['img'].values:
        print('not found: ', mask_)
    else:
        data_box_r1.loc[data_box_r1['img'] == mask_, 'img_path'] = path_dicom

# COMBINE DATASETS and SAVE
data_box_r123 = pd.concat([data_box_r1, data_box_r23], axis=0)
data_box_r123.to_excel('data/AIforCOVID/processed/box_data_AXF123.xlsx', index=False)

# Save R2-R3 bbox masks
data_box_r23.to_excel('data/AIforCOVID/processed/box_data_AXF23.xlsx', index=False)












print('here')


