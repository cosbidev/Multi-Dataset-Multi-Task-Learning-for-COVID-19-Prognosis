import os
import random
import cv2
import imutils
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision
from PIL import Image
from scipy.ndimage import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torch.utils.data import IterableDataset, DataLoader, Sampler
from tqdm import tqdm

from .utils_images import get_box, get_mask, normalize, find_bboxes
from .utils_visualization import plot_bbox_on_image


def create_combined_folds(cv_option, morbidity_cfg, severity_cfg, steps=None):
    if steps is None:
        steps = ['train', 'val', 'test']
    else:
        # Check steps is a list
        assert isinstance(steps, list)
    # Check cv_option is valid
    assert cv_option in ['loCo6', 'loCo18', 5]
    # Check morbidity_cfg and severity_cfg are dictionaries
    if cv_option == 'loCo6':
        fold_range_Morb = [('M' + value, int(value)) for value in os.listdir(morbidity_cfg['img']['fold_dir'])]
        fold_range_Sev = [('S' + value, int(value)) for value in os.listdir(severity_cfg['img']['fold_dir'])]
        fold_range_Sev = fold_range_Sev[:1]
        unique_combinations = []
        fold_id = []

        for i in range(len(fold_range_Morb)):
            for j in range(len(fold_range_Sev)):
                unique_combinations.append((fold_range_Morb[i], fold_range_Sev[j]))
                fold_id.append(fold_range_Morb[i][0] + '_' + fold_range_Sev[j][0])

        fold_grid = {fold: {} for fold in fold_id}

        for fold, id_modality in zip(fold_grid.keys(), unique_combinations):
            fold_data_combined = {step: None for step in steps}
            for step in steps:
                data_morbidity = pd.read_csv(os.path.join(morbidity_cfg['img']['fold_dir'], str(id_modality[0][1]), '%s.txt' % step), delimiter=" ")
                data_morbidity.loc[:, 'dataset_class'] = np.array(['AFC' for i in range(data_morbidity.shape[0])])

                data_severity = (pd.read_csv(os.path.join(severity_cfg['img']['fold_dir'], str(id_modality[1][1]), '%s.txt' % step), delimiter=" ")
                                 .drop(columns=['label_dim']).rename(columns={'scores': 'label'}))
                data_severity.loc[:, 'dataset_class'] = np.array(['BX' for i in range(data_severity.shape[0])])

                combined_dataset = pd.concat([data_morbidity, data_severity], axis=0).reset_index(drop=True)
                fold_data_combined[step] = combined_dataset
            fold_grid[fold] = fold_data_combined
        return fold_grid, fold_id
    elif cv_option == 'loCo18':
        fold_range_Morb = [('M' + value, int(value)) for value in os.listdir(morbidity_cfg['img']['fold_dir'])]
        fold_range_Sev = [('S' + value, int(value)) for value in os.listdir(severity_cfg['img']['fold_dir'])]

        unique_combinations = []
        fold_id = []

        for i in range(len(fold_range_Morb)):
            for j in range(len(fold_range_Sev)):
                unique_combinations.append((fold_range_Morb[i], fold_range_Sev[j]))
                fold_id.append(fold_range_Morb[i][0] + '_' + fold_range_Sev[j][0])

        fold_grid = {fold: {} for fold in fold_id}

        for fold, id_modality in zip(fold_grid.keys(), unique_combinations):
            fold_data_combined = {step: None for step in steps}
            for step in steps:
                data_morbidity = pd.read_csv(os.path.join(morbidity_cfg['img']['fold_dir'], str(id_modality[0][1]), '%s.txt' % step), delimiter=" ")
                data_morbidity.loc[:, 'dataset_class'] = np.array(['AFC' for i in range(data_morbidity.shape[0])])

                data_severity = (pd.read_csv(os.path.join(severity_cfg['img']['fold_dir'], str(id_modality[1][1]), '%s.txt' % step), delimiter=" ")
                                 .drop(columns=['label_dim']).rename(columns={'scores': 'label'}))
                data_severity.loc[:, 'dataset_class'] = np.array(['BX' for i in range(data_severity.shape[0])])

                combined_dataset = pd.concat([data_morbidity, data_severity], axis=0).reset_index(drop=True)
                fold_data_combined[step] = combined_dataset
            fold_grid[fold] = fold_data_combined
        return fold_grid, fold_id
    elif cv_option == 5:

        # Data Loaders for MORBIDITY TASK
        fold_grid = {fold: {} for fold in list(range(cv_option))}
        fold_id = list(range(cv_option))
        for fold in fold_grid.keys():

            fold_data_combined = {step: None for step in steps}
            for step in steps:
                data_morbidity = pd.read_csv(os.path.join(morbidity_cfg['img']['fold_dir'], str(fold), '%s.txt' % step), delimiter=" ")
                data_morbidity.loc[:, 'dataset_class'] = np.array(['AFC' for i in range(data_morbidity.shape[0])])

                data_severity = (pd.read_csv(os.path.join(severity_cfg['img']['fold_dir'], str(fold), '%s.txt' % step), delimiter=" ")
                                 .drop(columns=['label_dim']).rename(columns={'scores': 'label'}))
                data_severity.loc[:, 'dataset_class'] = np.array(['BX' for i in range(data_severity.shape[0])])

                combined_dataset = pd.concat([data_morbidity, data_severity], axis=0).reset_index(drop=True)
                fold_data_combined[step] = combined_dataset

            fold_grid[fold] = fold_data_combined
        return fold_grid, fold_id


def get_img_loader(loader_name):
    if loader_name == "custom_preprocessing":
        return loader
    else:
        raise ValueError(loader_name)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
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


"""class XRayResizer(object):
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

"""


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


def normalize_XRay(img, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    """Normalize using means and stds from ImageNet."""

    transform = torchvision.transforms.Normalize(means, stds)
    return transform(img)


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
    height, width = img.shape[:2]  # It's also the final desired shape
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


def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    max_over_all = 0.0
    min_over_all = 1000000000000000
    for img, (min_, max_) in tqdm(loader):
        if max_ > max_over_all:
            max_over_all = max_
        if min_ < min_over_all:
            min_over_all = min_
    return max_over_all, min_over_all


def augmentation(img):
    # shift
    r = random.randint(0, 100)
    if r > 70:
        shift_perc = 0.1
        r1 = random.randint(-int(shift_perc * img.shape[0]), int(shift_perc * img.shape[0]))
        r2 = random.randint(-int(shift_perc * img.shape[1]), int(shift_perc * img.shape[1]))
        img = shift(img, [r1, r2], mode='nearest')
    # zoom
    r = random.randint(0, 100)
    if r > 70:
        zoom_perc = 0.1
        zoom_factor = random.uniform(1 - zoom_perc, 1 + zoom_perc)
        img = clipped_zoom(img, zoom_factor=zoom_factor)
    # flip
    r = random.randint(0, 100)
    if r > 70:
        img = cv2.flip(img, 1)
    # rotation
    r = random.randint(0, 100)
    if r > 70:
        max_angle = 10
        r = random.randint(-max_angle, max_angle)
        img = imutils.rotate(img, r)
    # elastic deformation
    r = random.randint(0, 100)
    if r > 70:
        img = elastic_transform(img, alpha_range=[20, 40], sigma=7)
    return img


def loader(img_path, img_dim, masked=False, mask_path=None, bbox_resize=False, step="train", **kwargs):
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
    # Img Loading and selection
    img, photometric_interpretation = load_img(img_path)
    if 'BRIXIA' in mask_path:
        mask, _ = load_img(mask_path)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    else:
        mask, _ = load_img(mask_path)
    min_val, max_val = img.min(), img.max()
    if photometric_interpretation == 'MONOCHROME1':
        img = np.interp(img, (min_val, max_val), (max_val, min_val))
    # [MASKED IMAGE] calculate max and min on lung and clip the image
    masked_image = get_mask(img, mask)
    masked_image = np.where(masked_image != 0.0, masked_image, masked_image.mean())
    lower, upper = np.percentile(masked_image.flatten(), [2, 98])
    masked_image = np.clip(masked_image, lower, upper)
    min_val_lung, max_val_lung = masked_image.min(), masked_image.max()
    # Min max

    img = np.where(img != 0.0, img, img.mean())
    lower_img, upper_img = np.percentile(img.flatten(), [2, 98])
    img = np.clip(img, lower_img, upper_img)
    min_val, max_val = img.min(), img.max()
    # Select Box Area
    if masked:
        img = np.clip(img, min_val_lung, max_val_lung)
        img = get_mask(img, mask)
        min_val, max_val = img.min(), img.max()

    if kwargs['normalization_mode'] == 'MinMax':
        img = normalize(img, min_val=min_val, max_val=max_val)
    if bbox_resize:
        try:
            box_tot, _, _ = find_bboxes(mask)
            img = get_box(img=img, box_=box_tot, masked=masked, **kwargs)
        except IndexError:
            img = img
    # To Grayscale
    if img.ndim > 2:
        img = img.mean(axis=2)

    # Resize


    img = cv2.resize(img, (img_dim, img_dim))

    """    # Augmentation
    if step == "train":
        img = augmentation(img)

    # 3 channels
    img = np.stack((img,) * 3, axis=2).reshape(3, img_dim, img_dim)
    # To Tensor
    img = torch.Tensor(img.reshape(3, img_dim, img_dim) if img.shape[0] != 3 else img)"""
    return img


class CustomSampler(Sampler):
    def __init__(self, dataset1, dataset2, ratio, batch_size):

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.ratio_d1 = ratio

        self.batch_size = batch_size

        self.num_samples_dataset1 = int(batch_size * self.ratio_d1)
        self.num_samples_dataset2 = batch_size - self.num_samples_dataset1

        self.indices1 = np.arange(len(dataset1))
        self.indices2 = np.arange(len(dataset2)) + len(dataset1)

        np.random.shuffle(self.indices1)
        np.random.shuffle(self.indices2)

        self.current_idx1 = 0
        self.current_idx2 = 0

        self.max_batches = (len(dataset1) + len(dataset2)) // batch_size
    def __iter__(self):
        batch = []
        batch_count = 0
        while batch_count <= self.max_batches:
            for _ in range(self.num_samples_dataset1):
                if self.current_idx1 >= len(self.indices1):
                    np.random.shuffle(self.indices1)
                    self.current_idx1 = 0
                batch.append(self.indices1[self.current_idx1])
                self.current_idx1 += 1

            for _ in range(self.num_samples_dataset2):
                if self.current_idx2 >= len(self.indices2):
                    np.random.shuffle(self.indices2)
                    self.current_idx2 = 0
                batch.append(self.indices2[self.current_idx2])
                self.current_idx2 += 1

            if len(batch) == self.batch_size:
                yield batch
                batch = []
                batch_count += 1

    def __len__(self):
        total_batches = (len(self.dataset1) + len(self.dataset2)) // self.batch_size
        return total_batches * self.batch_size


class CustomIterableDataset(torch.utils.data.Dataset):

    def __init__(self, data, dataset):
        # Store the filename in object's memory
        self.data = data
        self.dataset = dataset

    def __len__(self):
        # return the total number of samples
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        # Map each element using the line_mapper
        id_ = str(row.img)
        x = self.dataset.load_single_image(id_)
        return str(id_), x


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, step):
        self.normalize_strategy = None
        self.img_dim = None
        self.cfg = cfg
        self.step = step
        self.data = None
        self.images = {}
        self.loader = loader
        self.masks = {}
        self.boxes = {}
        self.img_paths = {}

    def set_normalize_strategy(self, normalize_strategy):
        self.normalize_strategy = normalize_strategy

    def load_single_image(self, id_):
        id_ = str(id_)
        if self.masks:
            mask_path = self.masks[id_]
        else:
            mask_path = None
        # load box
        if self.boxes:
            box = self.boxes[id_]
        else:
            box = None
        # Load data and get label
        img_path = self.img_paths[id_]
        x = self.loader(img_path=img_path, img_dim=self.img_dim, mask_path=mask_path, box=box, step=self.step, normalization_mode = self.normalize_strategy, **self.cfg['preprocess'])
        return x

    def load_images(self):
        c = CustomIterableDataset(self.data, self)
        loader_ = torch.utils.data.DataLoader(c, batch_size=32, num_workers=6)
        for output in tqdm(loader_):
            ids = output[0]
            x = output[1]
            for id, img in zip(ids, x):
                self.images[id] = img.cpu().numpy()

    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def drop_patient(patient_ids, data):
        data = data[~data['img'].isin(patient_ids)]
        data = data.reset_index(drop=True)
        return data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def get_label(self, y, Id_, class_dataset=None):
        raise NotImplementedError("get_label method not implemented")

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        Id_ = str(row.img)

        x = self.images[str(Id_)] if not self.normalize_strategy == 'MinMax' else normalize(self.images[str(Id_)], min_val=self.images[str(Id_)].min(), max_val=self.images[str(
            Id_)].max())
        # Augmentation
        if self.step == "train":
            x = augmentation(x)

        # 3 channels
        x = np.stack((x,) * 3, axis=2)

        y = row.label if 'scores' not in row.index else None  # label
        # Get label
        class_dataset = row.dataset_class if 'dataset_class' in row.index else None
        y = self.get_label(y=y, Id_=Id_, class_dataset=class_dataset)
        # Normalize
        x = normalize_XRay(torch.Tensor(x.reshape(3, self.img_dim, self.img_dim))) if self.normalize_strategy == 'Imagenet' else torch.Tensor(x.reshape(
            3, self.img_dim, self.img_dim))
        if 'dataset_class' not in row.index:
            return x, y, Id_, 'NO_CLASS'
        else:
            return x, y, Id_, class_dataset


class DatasetImgAFC(BaseDataset):
    'Characterizes a dataset for PyTorch Dataloader to trait images'

    def __init__(self, data, classes, cfg, step, one_hot=True, load=True):
        """
        Initialization super class and specific attributes for the specific dataset.
        Args:
            data: dataframe with the data
            classes: list of classes
            cfg: configuration file
            step: step of the pipeline (train, valid, test)
            one_hot: if True, the label is converted to one hot encoding
            """
        BaseDataset.__init__(self, cfg, step)

        # Specific Attributes for the specific dataset
        self.m_idx_to_class = None
        self.m_class_to_idx = None
        self.class_to_one_hot = None
        self.one_hot = one_hot
        self.one_hot_list = list()
        self.load = load
        self.set_normalize_strategy(cfg['trainer']['normalizer'])


        if load:
            # Load information for Morbidity
            self.cfg = cfg['data']['modes']['img']
            self.img_dim = self.cfg['img_dim']
            self.classes = classes
            self.data, self.masks, self.boxes, self.box_R, self.box_L, self.img_paths = self.process_morbidity(self.cfg, data, classes)
            self.shuffle()
            # Processing and Loading of images in memory

            self.load_images()


    def process_morbidity(self, cfg, data, classes):
        data = self.drop_patient(data=data,
                                 patient_ids=['P_3_391', 'P_3_377', 'P_3_20', 'P_3_108', 'P_1_16', 'P_3_341', 'P_3_411.dcm',
                                              'P_3_208.dcm'])

        self.one_hot_list = [[1, 0], [0, 1]]
        self.class_to_one_hot = {c: self.one_hot_list[i] for i, c in enumerate(sorted(classes))}
        self.m_class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.m_idx_to_class = {i: c for c, i in self.m_class_to_idx.items()}
        cfg = cfg['img'] if not self.load else cfg
        # Mask (to select only the lungs pixels)
        if cfg['mask_dir']:
            masks = {str(id_patient): os.path.join(cfg['mask_dir'], '%s.tif' % id_patient) for id_patient in data['img']}
        else:
            masks = None
        # Box (to select only the box containing the lungs)
        if cfg['box_file']:
            box_data = pd.read_excel(cfg['box_file'], index_col="img", dtype=list)
            boxes = {str(row[0]): eval(row[1]["all"]) for row in box_data.iterrows()}
            box_R = {str(row[0]): eval(row[1]["dx"]) if isinstance(row[1]["dx"], str) else [] for row in
                     box_data.iterrows()}
            box_L = {str(row[0]): eval(row[1]["sx"]) if isinstance(row[1]["sx"], str) else [] for row in box_data.iterrows()}
            img_paths = {str(row[0]): row[1]["img_path"] for row in box_data.iterrows()}
        else:
            boxes = None

        return data, masks, boxes, box_R, box_L, img_paths

    def get_label(self, y, Id_, class_dataset=None):
        Id_ = None
        if self.one_hot:
            return torch.Tensor(self.class_to_one_hot[y])
        else:
            return torch.Tensor(y)


class DatasetImageCXR(torch.utils.data.Dataset):
    def __init__(self, data, classes, cfg, step):
        'Initialization'
        self.cfg = cfg
        self.step = step
        self.data = data
        self.only_Lung = True
        # self.data = self.drop_patient()
        # Mask (to select only the lungs pixels)
        if cfg['mask_dir']:
            self.masks = {id_patient: os.path.join(cfg['mask_dir'], '%s.tiff' % id_patient) for id_patient in data['img']}
        else:
            self.masks = None
        # Box (to select only the box containing the lungs)
        if cfg['box_file']:
            box_data = pd.read_excel(cfg['box_file'], index_col="img", dtype=list)
            self.boxes = {row[0]: eval(row[1]["all"]) for row in box_data.iterrows()}
            self.img_paths = {row[0]: os.path.join(self.cfg['img_dir'], 'dicom_clean', str(row[0]) + '.dcm') for row in box_data.iterrows()}
        else:
            self.boxes = None
        self.img_dim = cfg['img_dim']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.data.iloc[index]
        id = row.img
        img_path = self.img_paths[id]

        img, photometric_interpretation = load_img(img_path)
        if self.masks:
            mask, _ = load_img(self.masks[id])
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Obtain the image with only the lungs
        img = get_mask(img, mask)
        # Select Box Area

        if self.boxes:
            box_tot, _, _ = find_bboxes(mask)
            img = get_box(img=img, box_=box_tot, masked=False)

        img = np.where(img != 0.0, img, img.mean())
        img = np.where(img < 64000, img, img.mean())
        img = np.clip(img, 1500, img.max())
        min_val, max_val = img.min(), img.max()

        # Pathometric Interpretation
        if photometric_interpretation == 'MONOCHROME1':
            img = np.interp(img, (min_val, max_val), (max_val, min_val))
            min_val, max_val = img.min(), img.max()

        # Resize
        img = cv2.resize(img, (self.img_dim, self.img_dim))
        # 3 channels
        img = np.stack((img, img, img), axis=0)
        # To Tensor
        return img, (min_val, max_val)


class DatasetImgBX(BaseDataset):
    'Characterizes a dataset for PyTorch Dataloader to trait images'

    def __init__(self, data, classes, cfg, step, load=True):
        'Initialization'
        BaseDataset.__init__(self, cfg, step)



        # Specific Attributes for the specific dataset
        self.s_idx_to_class = None
        self.s_class_to_idx = None
        self.brixia_scores = None
        self.load = load
        self.set_normalize_strategy(cfg['trainer']['normalizer'])

        if load:

            # Load information for Severity
            self.classes = classes
            self.cfg = cfg['data']['modes']['img']
            self.img_dim = self.cfg['img_dim']
            self.data, self.brixia_score, self.masks, self.boxes, self.box_R, self.box_L, self.img_paths = self.process_severity(self.cfg, data, classes)
            self.shuffle()
            # Processing and Loading of images in memory
            self.load_images()

    def process_severity(self, cfg, data, classes):
        self.drop_patient(patient_ids=['1773596264454332092'], data=data)
        self.s_class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        self.s_idx_to_class = {i: c for c, i in self.s_class_to_idx.items()}
        cfg = cfg['img'] if not self.load else cfg
        # Mask (to select only the lungs pixels)
        if cfg['mask_dir']:
            masks = {str(id_patient): os.path.join(cfg['mask_dir'], '%s.tiff' % id_patient) for id_patient in data['img']}
        else:
            masks = None
        # Box (to select only the box containing the lungs)
        if cfg['box_file']:
            box_data = pd.read_excel(cfg['box_file'], index_col="img", dtype=list)
            boxes = {str(row[0]): eval(row[1]["all"]) for row in box_data.iterrows()}
            box_R = {str(row[0]): eval(row[1]["dx"]) if isinstance(row[1]["dx"], str) else [] for row in
                     box_data.iterrows()}
            box_L = {str(row[0]): eval(row[1]["sx"]) if isinstance(row[1]["sx"], str) else [] for row in box_data.iterrows()}
            img_paths = {str(row[0]): os.path.join(cfg['img_dir'], 'dicom_clean', str(row[0]) + '.dcm') for row in box_data.iterrows()}
        else:
            boxes = None

        # BRIXIA SCORES DICTIONARY
        Brixia_score = self.create_BX_scores_table(data, classes)
        return data, Brixia_score, masks, boxes, box_R, box_L, img_paths

    @staticmethod
    def create_BX_scores_table(data, classes):
        """
        Create a table with the Brixia scores for each patient.
        Returns:
            BX_Scores_table: table with the Brixia scores for each patient.
        """
        # Create table empty
        BX_Scores_table = pd.DataFrame(index=data['img'], columns=['A', 'B', 'C', 'D', 'E', 'F', 'BX_R', 'BX_L', 'BX_Total'])
        columns_scores = list(BX_Scores_table.columns)[:6]

        for patient_data in data.iterrows():
            brixia_score = patient_data[1]['scores'].replace('[', '').replace(']', '') if 'scores' in patient_data[1].index else patient_data[1]['label'].replace('[', '').replace(']', '')
            score_tot = {zone: 0 for i, zone in zip(range(classes.__len__()), columns_scores)}
            for j, score_bx in enumerate(brixia_score):
                score_tot[columns_scores[j]] = int(score_bx)
            # RIGHT LUNG SCORES:
            score_right = sum(list(score_tot.values())[0:3])
            # LEFT LUNG SCORES:
            score_left = sum(list(score_tot.values())[3:6])
            # TOTAL SCORES:
            score_total = sum(list(score_tot.values()))

            # Store scores in the table
            BX_Scores_table.loc[patient_data[1]['img']][columns_scores] = list(score_tot.values())
            BX_Scores_table.loc[patient_data[1]['img']]['BX_R'] = score_right
            BX_Scores_table.loc[patient_data[1]['img']]['BX_L'] = score_left
            BX_Scores_table.loc[patient_data[1]['img']]['BX_Total'] = score_total
        BX_Scores_table.index = BX_Scores_table.index.astype(str)
        return BX_Scores_table

    def get_label(self, Id_, class_dataset=None, y=None):

        return torch.Tensor(self.brixia_score.loc[Id_].array)


class MultiTaskDataset(DatasetImgBX, DatasetImgAFC):

    def __init__(self, data, cfg_morbidity, cfg_severity, step, cfg, one_hot=True):
        """

        :param data:
        :param cfg_morbidity:
        :param cfg_severity:
        :param step:
        :param cfg:
        :param one_hot:
        """
        # Initialization

        self.set_normalize_strategy(cfg['trainer']['normalizer'])
        cfg.img_dim = cfg['data']['img_dim']
        cfg.mask_dir = cfg_morbidity.img.mask_dir
        cfg.box_file = cfg_morbidity.img.box_file
        cfg.img_dir = cfg_morbidity.img.img_dir

        DatasetImgAFC.__init__(self, data=data, classes=cfg_morbidity['classes'], cfg=cfg, step=step, load=False)
        DatasetImgBX.__init__(self, data=data, classes=cfg_severity['classes'], cfg=cfg, step=step, load=False)

        self.morbidity_classes = cfg_morbidity['classes']
        self.severity_classes = cfg_severity['classes']
        self.cfg = cfg
        self.step = step
        self.data = data

        self.__s_class_to_idx = None
        self.m_idx_to_class = None
        self.__m_class_to_one_hot = None
        self.__m_class_to_idx = None
        self.__s_class_to_one_hot = None
        self.s_class_to_idx = None

        # Load information for Morbidity and Severity
        self.morbidity_data = self.data[self.data['dataset_class'] == 'AFC']
        self.severity_data = self.data[self.data['dataset_class'] == 'BX']

        self.__cfg_morbidity = cfg_morbidity
        self.__cfg_severity = cfg_severity

        self.__one_hot = one_hot
        self.img_dim = self.cfg['data']['img_dim']
        # Process Datasets
        self.morbidity_data, self.__m_masks, self.__m_boxes, self.__m_box_R, self.__m_box_L, self.__m_img_paths = self.process_morbidity(cfg_morbidity, self.morbidity_data,
                                                                                                                                         self.morbidity_classes)

        self.severity_data, self.brixia_scores, self.__s_masks, self.__s_boxes, self.__s_box_R, self.__s_box_L, self.__s_img_paths = self.process_severity(cfg_severity, self.severity_data,
                                                                                                                                                           self.severity_classes)

        self.data = pd.concat([self.morbidity_data, self.severity_data], axis=0).reset_index(drop=True)
        self.shuffle()

        # Specific Attributes for the specific dataset
        self.class_to_idx = {**self.m_class_to_idx, **self.s_class_to_idx}
        self.masks = {**self.__m_masks, **self.__s_masks}
        self.boxes = {**self.__m_boxes, **self.__s_boxes}
        self.box_R = {**self.__m_box_R, **self.__s_box_R}
        self.box_L = {**self.__m_box_L, **self.__s_box_L}
        self.img_paths = {**self.__m_img_paths, **self.__s_img_paths}
        self.load_images()

    def load_single_image(self, id_):
        id_ = str(id_)
        if self.masks:
            mask_path = self.masks[id_]
        else:
            mask_path = None
        # load box
        if self.boxes:
            box = self.boxes[id_]
        else:
            box = None
        # Load data and get label
        img_path = self.img_paths[id_]
        x = self.loader(img_path=img_path, img_dim=self.img_dim, mask_path=mask_path, box=box, step=self.step,  normalization_mode = self.normalize_strategy, **self.cfg['data'][
            'preprocess'])
        return x

    def get_label(self, Id_, class_dataset=None, y=None):
        if class_dataset == 'BX':
            y = self.brixia_scores.loc[Id_].array  # scores
        elif class_dataset == 'AFC':
            y = self.class_to_one_hot[y] + [-999 for i in range(7)]  # label + scores
        return torch.Tensor(y)
