import os
import random
import warnings
import cv2
import imutils
import math
import numpy as np
import pandas as pd
import pydicom
import skimage.transform
import torch
import torchvision
from PIL import Image
from scipy.ndimage import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from .utils_images import get_box, get_mask, normalize, PreprocessDicom, find_bboxes




def get_img_loader(loader_name):
    if loader_name == "custom_preprocessing":
        return loader
    elif loader_name == "XRayVision_preprocessing":
        return loader_XRayVision
    else:
        raise ValueError(loader_name)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_img(img_path):
    """
    Load image from path using the pydicom library .
    Args:
        img_path: Path of the image to load.

    Returns: Image as numpy array and photometric interpretation inside the dicom file for CXR.

    """
    filename, extension = os.path.splitext(img_path)
    if extension == ".dcm":
        dicom = pydicom.dcmread(img_path)
        img = dicom.pixel_array.astype(float)
        photometric_interpretation = dicom.PhotometricInterpretation
    else:
        img = Image.open(img_path)
        img = np.array(img).astype(float)
        photometric_interpretation = None
    return img, photometric_interpretation


class XRayResizer(object):
    def __init__(self, size, engine="skimage"):
        self.size = size
        self.engine = engine

    def __call__(self, img):
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant',
                                                preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            import cv2  # pip install opencv-python
            return cv2.resize(img[0, :, :],
                              (self.size, self.size),
                              interpolation=cv2.INTER_AREA
                              ).reshape(1, self.size, self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
def rotate(l, n):
    return l[n:] + l[:n]
class XRayCenterCrop(object):

    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)

def normalize_XRayVision(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


def clahe_transform(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply((img * 255).astype(np.uint8)) / 255
    return img


def elastic_transform(image, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

   # Arguments
       image: Numpy array with shape (height, width, channels).
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def augmentation(img):
    # shift
    r = random.randint(0, 100)
    if r > 70:
        shift_perc = 0.1
        r1 = random.randint(-int(shift_perc*img.shape[0]), int(shift_perc*img.shape[0]))
        r2 = random.randint(-int(shift_perc*img.shape[1]), int(shift_perc*img.shape[1]))
        img = shift(img, [r1, r2], mode='nearest')
    # zoom
    r = random.randint(0, 100)
    if r > 70:
        zoom_perc = 0.1
        zoom_factor = random.uniform(1-zoom_perc, 1+zoom_perc)
        img = clipped_zoom(img, zoom_factor=zoom_factor)
    # flip
    r = random.randint(0, 100)
    if r > 70:
        img = cv2.flip(img, 1)
    # rotation
    r = random.randint(0, 100)
    if r > 70:
        max_angle = 15
        r = random.randint(-max_angle, max_angle)
        img = imutils.rotate(img, r)
    # elastic deformation
    r = random.randint(0, 100)
    if r > 70:
        img = elastic_transform(img, alpha_range=[20, 40], sigma=7)
    return img


def loader(img_path, img_dim, masked=False, mask_path=None, bbox_resize=False, box=None, bbox_perc=False, step="train", **kwargs):
    """
    Load the CXR dicom from path and preprocess it.

    Args:
        img_path: the path of the CXR dicom
        img_dim: the dimension of the output image
        mask_path: the path of lungs' mask
        box: bounding box of the lungs
        step: step of the pipeline (train, valid, test)
        **kwargs: other parameters

    Returns:
        img: the preprocessed CXR

    """
    if 'P_38' in img_path:
        print(img_path)
    # Img
    img, photometric_interpretation = load_img(img_path)
    min_val, max_val = img.min(), img.max()
    # Pathometric Interpretation
    if photometric_interpretation == 'MONOCHROME1':
        img = np.interp(img, (min_val, max_val), (max_val, min_val))
        min_val, max_val = img.min(), img.max()
    # To Grayscale
    if img.ndim > 2:
        img = img.mean(axis=2)


    # Normalize
    img = normalize(img, min_val=min_val, max_val=max_val)
    # CLAHE / MEDIAN FILTER / CLIPPING
    img = PreprocessDicom(img, clip_limit=0.01, med_filt=3, **kwargs)
    # Filter Mask
    mask, _ = load_img(mask_path)
    if masked:

        img = get_mask(img, mask)

    # Select Box Area
    if bbox_resize:
        try:
            box_tot, _,_ = find_bboxes(mask)
            img = get_box(img, box_tot, masked=masked)
        except IndexError:
            print("error, box not found", img_path)
            img = img

    # Resize
    img = cv2.resize(img, (img_dim, img_dim))


    # Augmentation
    if step == "train":
        img = augmentation(img)

    # 3 channels
    img = np.stack((img, img, img), axis=0)
    # To Tensor
    img = torch.Tensor(img)
    return img


def loader_XRayVision(img_path, img_dim, mask_path=None, box=None, clahe=False, step="train"):
    # Img
    img, photometric_interpretation = load_img(img_path)
    min_val, max_val = img.min(), img.max()
    # Pathometric Interpretation
    if photometric_interpretation == 'MONOCHROME1':
        img = np.interp(img, (min_val, max_val), (max_val, min_val))
        min_val, max_val = img.min(), img.max()
    # Filter Mask
    if mask_path:
        mask, _ = load_img(mask_path)
        img = get_mask(img, mask, value=1)
    # Select Box Area
    # Normalize
    img = normalize_XRayVision(img, maxval=max_val)
    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")
    # Add color channel
    img = img[None, :, :]
    transform = torchvision.transforms.Compose([XRayCenterCrop(), XRayResizer(img_dim)])
    img = transform(img)
    # To Tensor
    img = torch.Tensor(img)
    return img


class DatasetImgAFC(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch Dataloader to trait images'
    def __init__(self, data, classes, cfg, step):
        'Initialization'
        self.cfg = cfg
        self.step = step
        self.data = data
        self.data = self.drop_patient(['P_38', 'P_389', 'P_385', 'P_382','P_386', 'P_381', 'P_3_391',
                                       'P_3_377','P_3_20', 'P_3_108', 'P_1_16', 'P_3_341', 'P_3_411.dcm', 'P_3_208.dcm'])
        self.shuffle()
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        # Mask (to select only the lungs pixels)
        if cfg['mask_dir']:
            self.masks = {id_patient: os.path.join(cfg['mask_dir'], '%s.tif' % id_patient) for id_patient in data['img']}
        else:
            self.masks = None
        # Box (to select only the box containing the lungs)
        if cfg['box_file']:
            box_data = pd.read_excel(cfg['box_file'], index_col="img", dtype=list)
            self.boxes = {row[0]: eval(row[1]["all"]) for row in box_data.iterrows()}
            self.box_R = {row[0]: eval(row[1]["dx"]) if isinstance(row[1]["dx"], str) else [] for row in
                          box_data.iterrows()}
            self.box_L = {row[0]: eval(row[1]["sx"]) if isinstance(row[1]["sx"], str) else [] for row in box_data.iterrows()}
            self.img_paths = {row[0]: row[1]["img_path"] for row in box_data.iterrows()}
        else:
            self.boxes = None
        self.img_dim = cfg['img_dim']
        self.loader = get_img_loader(cfg['loader_name'])
    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
    def drop_patient(self, patient_ids):
        self.data = self.data[~self.data['img'].isin(patient_ids)]
        self.data = self.data.reset_index(drop=True)
        return self.data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.img
        if self.masks:
            mask_path = self.masks[id]
        else:
            mask_path = None
        # load box
        if self.boxes:
            box = self.boxes[id]
        else:
            box = None
        # Load data and get label
        img_path = self.img_paths[id]
        x = self.loader(img_path=img_path, img_dim=self.img_dim, mask_path=mask_path, box=box, **self.cfg['preprocess'])
        y = row.label # label
        return x, self.class_to_idx[y], id


class DatasetClinical(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, classes, cfg_data, step):
        'Initialization'
        self.step = step  # train valid or test
        self.clinical_data = pd.read_csv(cfg_data['data_file'], index_col=0) # clinical data
        self.data = data  # patient_id per fold with associated class
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        # Load data and get label
        x = torch.tensor(self.clinical_data.loc[id].astype(float))
        y = row.label
        return x, self.class_to_idx[y], id


class MultimodalDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset_mode_1, dataset_mode_2, classes):
        'Initialization'
        self.dataset_mode_1 = dataset_mode_1
        self.dataset_mode_2 = dataset_mode_2
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset_mode_1)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x1, label, idx = self.dataset_mode_1[index]
        x2, label, idx = self.dataset_mode_2[index]
        return x1, x2, label, idx


class FeatureDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, classes, features_file, step):
        'Initialization'
        self.step = step  # train valid or test
        self.features_data = pd.read_csv(features_file, index_col=0)
        self.data = data  # patient_id per fold with associated class
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        # Load data and get label
        x = torch.tensor(self.features_data.loc[id].astype(float))
        y = row.label
        return x, self.class_to_idx[y], id

class PrototypeDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, labels, patients_id):
        'Initialization'
        self.data = data  # patient_id per fold with associated class
        self.labels = labels
        self.idp = patients_id
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)


    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        idp = self.idp[index]
        return x, y, idp
