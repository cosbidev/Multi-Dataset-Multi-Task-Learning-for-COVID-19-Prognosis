import os
import random

import cv2
import imutils
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from scipy.ndimage import shift, gaussian_filter, map_coordinates


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
        max_angle = 15
        r = random.randint(-max_angle, max_angle)
        img = imutils.rotate(img, r)
    # elastic deformation
    r = random.randint(0, 100)
    if r > 70:
        img = elastic_transform(img, alpha_range=[20, 40], sigma=7)
    return img
def find_max_left_right(bboxes):
    # This function extract the two biggest boxes
    areas = [elem[1] for elem in bboxes]
    boxes_ = [elem[0] for elem in bboxes]
    if len(boxes_) < 2:
        bbox_tot = boxes_[0]
        return bbox_tot, None, None
    else:
        two_boxes = [boxes_[i] for i in list(np.argsort(areas, axis=0)[-2:])]
        # MAX BOX TOTAL
        matrix = np.zeros((2,len(two_boxes[0])))
        for i, bbox in enumerate(two_boxes):
            x, y, w, h = bbox
            matrix[i,0] = x
            matrix[i,1] = x + w
            matrix[i,2] = y
            matrix[i,3] = y + h
        bbox_tot = [np.min(matrix[:,0]), np.min(matrix[:,2]), np.max(matrix[:,1]) - np.min(matrix[:,0]), np.max(matrix[:,3]) - np.min(matrix[:,2])]
        bbox_left = two_boxes[np.argmin(matrix[:,0])]
        bbox_right = two_boxes[np.argmax(matrix[:,0])]
        return bbox_tot, bbox_left, bbox_right
def find_bboxes(mask):
    # get contours
    contours = cv2.findContours(mask.astype(np.int32).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    bboxes = []
    for cntr in contours:
        bbox = cv2.boundingRect(cntr)
        area = bbox[2] * bbox[3]
        bboxes.append((list(bbox), area))
    max_bbox, left_bbox, right_bbox = find_max_left_right(bboxes)
    return max_bbox, left_bbox, right_bbox

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
def PreprocessDicom(img, clip_limit=0.01, med_filt=3, clahe=False, filter=False, clip=False, **kwargs):
    """
    Preprocess single CXR with clahe, median filtering and clipping
    :param img: input image (Rows, Cols)
    :param clip_limit: CLAHE clip limit
    :param med_filt: median filter kernel size
    :return: (Rows, Cols)
    """
    img = img.astype('float32')/img.max()

    lower, upper = np.percentile(img.flatten(), [2, 98])
    if clip:
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)
    return img




def get_mask(img, mask):  # todo: img[~mask]
    boolean_mask = mask.astype(bool)
    return img * boolean_mask


def get_box(img, box_, masked=False, border_add=10, **kwargs):
    """
    Returns the image inside the bounding box, if the parameter masked is true the image is padded with zeros; otherwise
    it's needed to pad the image with real values from the image. The padding is done in order to have a squared image
    Args:
        img: ndarray of shape (H, W)
        box_: list of 4 values [x, y, w, h]
        masked: ndarray of shape (H, W), type bool

    Returns:
        the img padded
    """

    # BBOX parameters = [x, y, w, h]
    box = [int(b) for b in box_]
    # Sides
    if border_add == 0:
        pass
    else:
        pixel_add_w = ((box[2] * border_add) // 100) // 2
        pixel_add_h = ((box[3] * border_add) // 100) // 2

        if box[0] - pixel_add_w < 0:
            pixel_add_w = pixel_add_w - abs(box[0] - pixel_add_w)
        if box[0] + box[2] + pixel_add_w > img.shape[1]:
            pixel_add_w = pixel_add_w - abs(box[0] + box[2] + pixel_add_w - img.shape[1])
        if box[1] - pixel_add_h < 0:
            pixel_add_h = pixel_add_h - abs(box[1] - pixel_add_h)
        if box[1] + box[3] + pixel_add_h > img.shape[0]:
            pixel_add_h = pixel_add_h - abs(box[1] + box[3] + pixel_add_h - img.shape[0])

        box = [box[0] - pixel_add_w, box[1] - pixel_add_h, box[2] + pixel_add_w, box[3] + pixel_add_h]

    l_w = box[2]
    l_h = box[3]

    # Img dims
    img_w = img.shape[1]
    img_h = img.shape[0]
    rigth_top = int(np.floor(abs(l_w - l_h) / 2) + 1) if abs(l_w - l_h) % 2 != 0 else int(np.floor(abs(l_w - l_h) / 2))
    left_down = int(np.floor(abs(l_w - l_h) / 2)) if abs(l_w - l_h) % 2 != 0 else int(np.floor(abs(l_w - l_h) / 2))
    if masked:
        # Handle masked image
        img_to_box = img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
        # Padding with zeros
        if l_w < l_h:
            img_pad = np.pad(img_to_box, ((0, 0), (int(left_down), int(rigth_top))), constant_values=(0, 0))
        elif l_h < l_w:
            img_pad = np.pad(img_to_box, ((int(left_down), int(rigth_top)), (0, 0)), constant_values=(0, 0))
        else:
            return img_to_box
        return img_pad

    else:
        boolean_condition = True
        if l_w < l_h:
            # Padding with real values
            if box[0] < left_down:
                shift = abs(box[0] - left_down)
                left_down = int(left_down - shift)
                rigth_top = int(rigth_top + shift)
            elif box[0] + box[2] + rigth_top > img_w:
                shift = abs(box[0] + box[2] + rigth_top - img_h)
                left_down = int(left_down + shift)
                rigth_top = int(rigth_top - shift)
            if box[0] - left_down < 0:
                return img[box[1]: box[1] + box[3], : box[0] + box[2] + rigth_top]
            elif box[0] + box[2] + rigth_top > img_w:
                return img[box[1]: box[1] + box[3], box[0] - left_down:]
            else:
                return img[box[1]: box[1] + box[3], box[0] - left_down: box[0] + box[2] + rigth_top]

        elif l_h < l_w:
            if box[1] < left_down:
                shift = abs(box[1] - left_down)
                left_down = int(left_down - shift)
                rigth_top = int(rigth_top + shift)
            elif box[1] + box[3] + rigth_top > img_h:
                shift = abs(box[1] + box[3] + rigth_top - img_h)
                left_down = int(left_down + shift)
                rigth_top = int(rigth_top - shift)

            if box[1] - left_down < 0:
                return img[: box[1] + box[3] + rigth_top, box[0]: box[0] + box[2]]
            elif box[1] + box[3] + rigth_top > img_h:
                return img[box[1] - left_down:, box[0]: box[0] + box[2]]
            else:
                return img[box[1] - left_down: box[1] + box[3] + rigth_top, box[0]: box[0] + box[2]]
        elif l_h == l_w:
            return img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]

        # Padding with real values


def normalize(img, min_val=None, max_val=None):
    if not min_val:
        min_val = img.min()
    if not max_val:
        max_val = img.max()
    img = (img - min_val) / (max_val - min_val)
    # img -= img.mean()
    # img /= img.std()
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
    # Img

    img, photometric_interpretation = load_img(img_path)



    min_val, max_val = img.min(), img.max()
    # FInd
    img = np.where(img != 0.0, img, max_val)
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
    if 'BRIXIA' in mask_path:
        mask, _ = load_img(mask_path)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    else:
        mask, _ = load_img(mask_path)

    if masked:
        img = get_mask(img, mask)

    # Select Box Area
    if bbox_resize:
        try:

            box_tot, _, _ = find_bboxes(mask)
            img = get_box(img=img, box_=box_tot, masked=masked, **kwargs)

        except IndexError:
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



class DatasetImgAFC(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch Dataloader to trait images'
    def __init__(self, data, classes, cfg, step, one_hot=True):
        'Initialization'
        self.cfg = cfg
        self.step = step
        self.data = data
        self.data = self.drop_patient([ 'P_3_391', 'P_3_377','P_3_20', 'P_3_108', 'P_1_16', 'P_3_341', 'P_3_411.dcm',
                                        'P_3_208.dcm'])
        self.shuffle()
        self.one_hot = one_hot
        self.classes = classes
        self.one_hot_list = [[1, 0], [0, 1]]
        self.class_to_one_hot = {c: self.one_hot_list[i] for i, c in enumerate(sorted(classes))}
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
        self.loader = loader

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
        x = self.loader(img_path=img_path, img_dim=self.img_dim, mask_path=mask_path, box=box, step=self.step, **self.cfg['preprocess'])
        y = row.label  # label

        if self.one_hot:
            y = torch.Tensor(self.class_to_one_hot[y])

        return x, y, id
