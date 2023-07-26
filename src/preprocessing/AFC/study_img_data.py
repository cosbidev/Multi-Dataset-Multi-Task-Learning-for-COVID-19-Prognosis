# Script to understand img dimension after segmentation_brixia
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import src.utils_data


# Dirs and Files
data_dir = "../data/AIforCOVID"
img_dir = os.path.join(data_dir, "imgs")
mask_dir = os.path.join(data_dir, "masks")
clinical_data_path = os.path.join(data_dir, "AIforCOVID.xlsx")
bounding_box_file = os.path.join("./data/processed", "box_data.xlsx")

# Load clinical data
clinical_data = pd.read_excel(clinical_data_path, index_col="ImageFile")

# Box and Mask
masks = {id_patient: os.path.join(mask_dir, '%s.tif' % id_patient) for id_patient in clinical_data.index}
box_data = pd.read_excel(bounding_box_file, index_col="id", dtype=list)
boxes = {row[0]: eval(row[1]["box"]) for row in box_data.iterrows()}

img_dims = {}
for i, id_patients in tqdm(enumerate(clinical_data.index)):
    img_path = os.path.join(img_dir, '%s.dcm' % id_patients)
    mask_path = masks[id_patients]
    box = boxes[id_patients]

    # Img
    img, photometric_interpretation = util_data.load_img(img_path)
    min_val, max_val = img.min(), img.max()
    # Pathometric Interpretation
    if photometric_interpretation == 'MONOCHROME1':
        img = np.interp(img, (min_val, max_val), (max_val, min_val))
    # To Grayscale
    if img.ndim > 2:
        img = img.mean(axis=2)
    # Filter Mask
    mask, _ = util_data.load_img(mask_path)
    img = util_data.get_mask(img, mask, value=1)
    # Select Box Area
    img = util_data.get_box(img, box, perc_border=0.5)

    # Dim
    img_dims[id_patients] = img.shape[0]

# PLot Histogram
img_dims = pd.Series(img_dims)
sns.histplot(x=img_dims, kde=True)
plt.show()