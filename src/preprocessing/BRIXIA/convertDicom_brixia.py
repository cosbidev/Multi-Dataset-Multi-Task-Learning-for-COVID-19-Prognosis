import os
import pydicom as dicom
import numpy as np
import pandas as pd
from PIL import Image
import PIL.ImageOps
from torchvision import transforms as torch_transforms
from multiprocessing import Pool
from functools import partial


def convert_dicom_to_image_BRIXIA(image, meta_data, path_to_data, transform=None, extension='.jpg'):
    """ Preprocesses a dicom image and saves it to disk as jpg

    Args:
        image (str): Name of the image to process
        path_to_data (str): Path to the dicom images
        meta_data (pd.Dataframe): Dataframe containing meta information of BrixIA
        transform (torchvision.Transforms): Torchvision transform object
        extension (str): Extension of the image to save
    """

    dcm = dicom.dcmread(os.path.join(path_to_data, image))
    img_array = dcm.pixel_array.astype(np.float32)
    min_val, max_val = img_array.min(), img_array.max()
    """   
    THIS PART IS LOOSING INFORMATION ABOUT DATA DISTRIBUTION! 
    max_gray = np.max(img_array)
    
    # Scale 16-bit gray values of dicom images
    if max_gray <= 4095:
        img_array = (img_array/4095*255).astype(np.uint8)
    else:
        img_array = (img_array/65535*255).astype(np.uint8)"""

    interpretation = meta_data.loc[image]['PhotometricInterpretation']
    if interpretation == 'MONOCHROME1':
        img = np.interp(img_array, (min_val, max_val), (max_val, min_val))
        min_val, max_val = img.min(), img.max()
    image = image.replace('.dcm', extension)
    """if img.ndim > 2:
        img = img.mean(axis=2)"""
    dest_path = path_to_data.replace('dicom_clean', 'images')
    os.makedirs(dest_path, exist_ok=True)

    # Transaform to Pil
    img_pil = Image.fromarray(img_array)

    img_pil.save(os.path.join(dest_path, image))


if __name__ == '__main__':

    print('Started preprossesing of BrixIA')

    base_data_folder = '../../../data/BRIXIA'

    path_to_data = os.path.join(base_data_folder ,'dicom_clean')
    meta_path = os.path.join(base_data_folder, 'metadata_global_v2.csv')
    meta_data = pd.read_csv(meta_path, sep=';', dtype={'BrixiaScore': str}, index_col='Filename')

    processes = 4
    size = (512, 512)

    transforms = torch_transforms = torch_transforms.Compose([
        torch_transforms.Resize(size)
    ])

    images_list = os.listdir(path_to_data)

    pool = Pool(processes=processes)

    wrapper = partial(convert_dicom_to_image_BRIXIA,
                      meta_data=meta_data,
                      path_to_data=path_to_data,
                      transform=transforms,
                      extension='.tiff')

    result = pool.map_async(wrapper, images_list)
    result.get()

    print('Finished preprocessing.')
