import os
import PIL.ImageOps
import cv2
import math
import numpy as np
import pydicom as dicom
import torch
from PIL import Image
from skimage import morphology, color, transform, filters
from skimage.exposure import equalize_adapthist

from .utils_path import files_in_folder



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




def PreprocessDicom(img, clip_limit=0.01, med_filt=3, clahe=False, filter=False, clip=False, **kwargs):
    """
    Preprocess single CXR with clahe, median filtering and clipping
    :param img: input image (Rows, Cols)
    :param clip_limit: CLAHE clip limit
    :param med_filt: median filter kernel size
    :return: (Rows, Cols)
    """
    img = img.astype('float32')/img.max()
    if clahe:
        img = equalize_adapthist(
            img, clip_limit=clip_limit)
    if filter:
        img = filters.median(img, np.ones(
            (med_filt,med_filt))).astype(np.float32)
    lower, upper = np.percentile(img.flatten(), [2, 98])
    if clip:
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)
    return img



def convert_dicom_to_image_BRIXIA(image, path_to_data, transforms):
    """ Preprocesses a dicom image and saves it to disk as jpg

    Args:
        image (str): Name of the image to process
        path_to_data (str): Path to the dicom images
        transforms (torchvision.Transforms): Torchvision transform object
        extension (str): Extension of the image to save
    """

    dcm = dicom.dcmread(os.path.join(path_to_data, image))
    img_array = dcm.pixel_array.astype(np.float32)
    min_val, max_val = img_array.min(), img_array.max()
    """   
    !!!BE-CAREFUL: THIS PART IS LOOSING INFORMATION ABOUT DATA DISTRIBUTION! 
    max_gray = np.max(img_array)

    # Scale 16-bit gray values of dicom images
    if max_gray <= 4095:
        img_array = (img_array/4095*255).astype(np.uint8)
    else:
        img_array = (img_array/65535*255).astype(np.uint8)
    # --: Dont use this code by now    
        """
    # Dicom metadatas
    interpretation = dcm.PhotometricInterpretation
    if interpretation == 'MONOCHROME1':
        img_array = np.interp(img_array, (min_val, max_val), (max_val, min_val))
        min_val, max_val = img_array.min(), img_array.max()

    if img_array.ndim > 2:
        img_array = img_array.mean(axis=2)

    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    img_array = Image.fromarray(img_array)
    original_shape = img_array.size
    if transform is not None:

        img_array = transforms(img_array)

    img_array = np.array(img_array)
    img_array = np.expand_dims(img_array, -1)
    img_array = normalize(img_array, min_val=min_val, max_val=max_val)
    """    img_array -= img_array.mean()
    img_array /= img_array.std()"""

    return img_array, original_shape
def Volume_mask_and_or(mask_1, mask_2 ,OR=True):
    mask_1 = mask_1.astype(bool)
    mask_2 = mask_2.astype(bool)
    shapes = mask_1.shape
    final_mask = np.zeros(shapes)
    function_booleans = {True: np.bitwise_or, False: np.bitwise_and}
    for k in range(shapes[2]):
        for j in range(shapes[1]):
            final_mask[:, j, k] = function_booleans[OR](mask_1[:, j, k], mask_2[:, j, k])
    print('here')

    return final_mask

def save_mask(mask, path, name):
    """Save the mask in the specified path
    """
    # Save the mask
    mask = PIL.Image.fromarray(mask)
    mask.save(os.path.join(path, name))
def plot_img(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def loadData(im_shape, img_dirs = None, img_dir = None, reduction = True):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    if img_dirs is not None:
        files_ext = []
        for img_dir in img_dirs:
            files_ext.extend(files_in_folder(img_dir, ".dcm"))
    elif img_dir is not None:
        files_ext = files_in_folder(img_dir, ".dcm")
    else:
        raise ValueError("img_dirs and img_dir cannot be both None")

    X = []
    or_shapes = []
    imgs_name = []

    for item in files_ext[:10]:
        try:
            dicom = dicom.dcmread(item)
            img = dicom.pixel_array
            min_val, max_val = img.min(), img.max()

            interpretation = dicom.PhotometricInterpretation
            img_pil = Image.fromarray(img)
            if interpretation == 'MONOCHROME1':

                img = np.interp(img, (min_val, max_val), (max_val, min_val))
                min_val, max_val = img.min(), img.max()

            img = np.array(img)
            imgs_name.append(item)
            or_shapes.append(img.shape)
            img = transform.resize(img, im_shape)
            """if len(img.shape) == 3:
                img = np.mean(img, axis=2)"""
            img = np.expand_dims(img, -1)
            """            img -= img.mean()
            img /= img.std()"""
            X.append(img)
        except RuntimeError as rerr:
            print("Error reading file {}".format(item), rerr.with_traceback())
            continue
    # Arrays of images (N, H, W, C)
    X = np.array(X)
    print('### Data loaded')
    print('\t{}'.format(X.shape))
    return X, or_shapes, imgs_name


def get_mask(img, mask): # todo: img[~mask]
    boolean_mask = mask.astype(bool)
    return img * boolean_mask


def get_box(img, box, masked=False):
    """
    Returns the image inside the bounding box, if the parameter masked is true the image is padded with zeros; otherwise
    it's needed to pad the image with real values from the image. The padding is done in order to have a squared image
    Args:
        img: ndarray of shape (H, W)
        box: list of 4 values [x, y, w, h]
        masked: ndarray of shape (H, W), type bool

    Returns:
        the img padded
    """
    # BBOX parameters = [x, y, w, h]
    box = [int(b) for b in box]
    # Sides
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
                return img[box[1] - left_down: , box[0]: box[0] + box[2]]
            else:
                return img[box[1] - left_down: box[1] + box[3] + rigth_top, box[0]: box[0] + box[2]]




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


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


def masked_pred(img, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [0, 0, 1]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


