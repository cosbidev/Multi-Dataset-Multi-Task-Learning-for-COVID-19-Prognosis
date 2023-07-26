import os
import pydicom as dicom
import numpy as np
import pandas as pd
from PIL import Image
from pydicom.errors import InvalidDicomError
from torchvision import transforms as torch_transforms
from src.utils import normalize
import click

def create_label_clinical_files(meta_data_global, output_dir, extension='csv'):
    """ Creates a label file from the meta data
    """
    # Output path
    print('Creating label file...')



    dest_path = os.path.join(os.path.dirname(output_dir), 'images')
    pass
def convert_dicom_to_image_AIforCovid(image_PATH,
                                      transform=None,
                                      extension='.jpg',
                                      saved_patient=pd.DataFrame(),
                                      output_dir=None):
    """ Preprocesses a dicom image and saves it to disk as jpg

    Args:
        path_to_data (str): Path to the dicom images
        meta_data (pd.Dataframe): Dataframe containing meta information of BrixIA
        transform (torchvision.Transforms): Torchvision transform object
        extension (str): Extension of the image to save
    """

    dcm = dicom.dcmread(image_PATH)
    print(image_PATH)
    try:
        img_array = dcm.pixel_array.astype(float)

        min_val, max_val = img_array.min(), img_array.max()

        """        # Scale 16-bit gray values of dicom images
        if max_gray <= 4095:
            img_array = (img_array / 4095 * 255).astype(np.uint8)
        else:
            img_array = (img_array / 65535 * 255).astype(np.uint8)"""
        img_pil = Image.fromarray(img_array)
        interpretation = dcm.PhotometricInterpretation
        if interpretation == 'MONOCHROME1':
            img = np.interp(img_array, (min_val, max_val), (max_val, min_val))
            min_val, max_val = img.min(), img.max()
        # Img original path
        original_path = image_PATH
        if img.ndim > 2:
            img = img.mean(axis=2)
        if transform is not None:
            img_pil = transform(img)
        # Normalize
        img = normalize(img, min_val=min_val, max_val=max_val)
        # Saving the new image
        patient_id_ext = image_PATH.split('/')[-1].replace('.dcm', extension)

        if output_dir is None:
            dest_path = os.path.join(os.path.dirname(os.path.dirname(image_PATH)), 'images')
        else:
            dest_path = os.path.join(os.path.dirname(output_dir), 'images')
        # CREATE FOLDER IF NOT EXIST
        os.makedirs(dest_path, exist_ok=True)
        path_saving = os.path.join(dest_path, patient_id_ext)
        saved_patient.loc[len(saved_patient)] = [patient_id_ext.split('.')[0], path_saving, original_path]
        img_pil.save(path_saving)
    except AttributeError as aerror:
        print(aerror)
        pass
    except RuntimeError as rerror:
        print(rerror)
        pass
    except InvalidDicomError as error:
        print(error)
        pass
    except ValueError as verror:
        print(verror)
        pass
    return saved_patient
@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--output_dir', help='Directory for the output dataset', default='/data/processed', metavar='PATH')



def preprocess_AIforCovid(**kwargs):
    """ Preprocesses the BrixIA dataset and saves it to disk
    """
    opt = dict(**kwargs)
    print('Started preprossesing of BrixIA')
    base_data_folder = 'data/AIforCOVID'
    path_to_data_1 = os.path.join(base_data_folder, 'imgs')
    path_to_data_2 = os.path.join(base_data_folder, 'imgs_r2')
    path_to_data_3 = os.path.join(base_data_folder, 'imgs_r3')
    # CLINICAL DATA
    meta_path = os.path.join(base_data_folder, 'AIforCOVID.xlsx')
    meta_path_2 = os.path.join(base_data_folder, 'AIforCOVID_r2.xlsx')
    meta_path_3 = os.path.join(base_data_folder, 'AIforCOVID_r3.xlsx')
    """    size = (512, 512)

    transforms = torch_transforms = torch_transforms.Compose([
        torch_transforms.Resize(size)
    ])"""
    # Clinical Data:
    clinical_meta_ = pd.read_excel(meta_path)
    clinical_meta_2 = pd.read_excel(meta_path_2)
    clinical_meta_3 = pd.read_excel(meta_path_3)
    clinical_meta_global = pd.concat([clinical_meta_, clinical_meta_2, clinical_meta_3])
    # Images:
    images_list_1 = [os.path.join(path_to_data_1, image_filename) for image_filename in os.listdir(path_to_data_1)]
    images_list_2 = [os.path.join(path_to_data_2, image_filename) for image_filename in os.listdir(path_to_data_2)]
    images_list_3 = [os.path.join(path_to_data_3, image_filename) for image_filename in os.listdir(path_to_data_3)]
    images_list = images_list_1 + images_list_2 + images_list_3
    images_list = sorted(images_list)
    saved_patient = pd.DataFrame(columns=['ID', 'Path', 'origin_Path'])
    for image in images_list:
        saved_patient = convert_dicom_to_image_AIforCovid(image_PATH=image,
                                                          transform=None,
                                                          extension='.tiff',
                                                          output_dir=opt['output_dir'],
                                                          saved_patient=saved_patient)
    # Label file creation
    label_file, clinical_data = create_label_clinical_files(meta_data_global=clinical_meta_global, output_dir=opt['output_dir'])
    print('Finished preprocessing.')

if __name__ == '__main__':
    preprocess_AIforCovid()
