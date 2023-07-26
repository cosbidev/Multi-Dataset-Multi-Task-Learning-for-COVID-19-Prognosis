# Many of the functions below are adapted from tesorflow-addons
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
from skimage import exposure, filters

def load_image(filename, target_size=512):
    """
    Load and resize a grayscale image from file
    :param filename: filename with path
    :param target_size: output size
    :return: loaded image as (num_rows, num_columns, 1)
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"File error: {filename}")
    img = img / 255
    img = cv2.resize(img, (target_size, target_size))
    img = np.reshape(img, img.shape + (1,))
    return img


def image_preprocess(img, clip_limit=0.01, med_filt=3):
    """
    preprocess input image
    :param med_filt: median filter kernel size
    :param clip_limit: CLAHE clip limit
    :param img: (Rows, Cols, 1)
    :return: (Rows, Cols, 1)
    """
    img = img[:,:,0]
    img = preprocess(img, clip_limit=0.01, med_filt=3)
    return np.expand_dims(img, axis=-1)


def preprocess(img, clip_limit=0.01, med_filt=3):
    """
    Preprocess single CXR with clahe, median filtering and clipping
    :param img: input image (Rows, Cols)
    :param clip_limit: CLAHE clip limit
    :param med_filt: median filter kernel size
    :return: (Rows, Cols)
    """
    img = img.astype('float32')/img.max()

    img_eq = exposure.equalize_adapthist(
        img, clip_limit=clip_limit)
    img_eq_median = filters.median(img_eq, np.ones(
        (med_filt,med_filt))).astype(np.float32)

    lower, upper = np.percentile(img_eq_median.flatten(), [2, 98])
    img_clip = np.clip(img_eq_median, lower, upper)
    return (img_clip - lower)/(upper - lower)


def add_suffix(base_file, suffix):
    if isinstance(base_file, str):
        filename, fileext = os.path.splitext(base_file)
        return "%s_%s%s" % (filename, suffix, fileext)
    elif isinstance(base_file, list):
        out = []
        for bf in base_file:
            filename, fileext = os.path.splitext(bf)
            out.append("%s_%s%s" % (filename, suffix, fileext))
        return out
    else:
        raise ValueError
