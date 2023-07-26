import os

import imutils
import pandas as pd
import skimage
from skimage import measure
from skimage.morphology import convex_hull_image
import cv2
from src import preprocess, AlignerSegmenter, loadData, remove_small_regions, save_mask, convert_dicom_to_image_BRIXIA, \
    Volume_mask_and_or
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as torch_transforms
"""def K_linspace(start, stop, num):
    return tf.linspace(start, stop, num)

def K_meshgrid(x, y):
    return tf.meshgrid(x, y)
"""

import numpy as np

def plot_image(image, title=None, show=True):
    plt.imshow(image, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()
class BilinearInterpolation(object):
    """Performs bilinear interpolation as a numpy layer"""

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size

    def get_config(self):
        return {
            'output_size': self.output_size,
        }

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):
        batch_size = np.shape(image)[0]
        height = np.shape(image)[1]
        width = np.shape(image)[2]
        num_channels = np.shape(image)[3]

        x = np.cast[np.float32](np.ravel(sampled_grids[:, 0:1, :]))
        y = np.cast[np.float32](np.ravel(sampled_grids[:, 1:2, :]))

        x = .5 * (x + 1.0) * np.cast[np.float32](width)
        y = .5 * (y + 1.0) * np.cast[np.float32](height)

        x0 = x.astype('int32')
        x1 = x0 + 1
        y0 = y.astype('int32')
        y1 = y0 + 1

        max_x = int(np.shape(image)[2] - 1)
        max_y = int(np.shape(image)[1] - 1)

        x0 = np.clip(x0, 0, max_x)
        x1 = np.clip(x1, 0, max_x)
        y0 = np.clip(y0, 0, max_y)
        y1 = np.clip(y1, 0, max_y)

        pixels_batch = np.arange(0, batch_size) * (height * width)
        pixels_batch = np.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = np.repeat(pixels_batch, flat_output_size, axis=1)
        base = np.ravel(base)

        base_y0 = y0 * width
        base_y0 = base + base_y0

        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = np.reshape(image, newshape=(-1, num_channels))
        flat_image = np.cast[np.float32](flat_image)
        pixel_values_a = flat_image[indices_a]
        pixel_values_b = flat_image[indices_b]
        pixel_values_c = flat_image[indices_c]
        pixel_values_d = flat_image[indices_d]

        x0 = np.cast[np.float32](x0)
        x1 = np.cast[np.float32](x1)
        y0 = np.cast[np.float32](y0)
        y1 = np.cast[np.float32](y1)

        area_a = np.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = np.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = np.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = np.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        x_linspace = np.linspace(-1., 1., width)
        y_linspace = np.linspace(-1., 1., height)
        x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
        x_coordinates = np.ravel(x_coordinates)
        y_coordinates = np.ravel(y_coordinates)
        ones = np.ones_like(x_coordinates)
        grid = np.concatenate([x_coordinates, y_coordinates, ones], 0)

        grid = np.ravel(grid)
        grids = np.tile(grid, (batch_size,))
        return np.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = np.shape(X)[0], np.shape(X)[3]
        transformations = np.reshape(affine_transformation, newshape=(batch_size, 2, 3))
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = np.matmul(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = np.reshape(interpolated_image, new_shape)
        return interpolated_image, affine_transformation



def create_mask(images_list,metadata,  Segmenter, transforms, mask_aligner, lung_aligner, path_to_data, alignment=False, threshold=0.50, dataset= 'AIforCovid', output_dir=''):
    """
    Preprocesses an image from the dataset and generates the mask for the lungs.
    Args:
        threshold: segmentation_brixia threshold for boolean mask
        Segmenter: Segmenter model
        transforms: apply transformation to the image if not None
        alignment: Alignment model
        path_to_data (str): Path to the dicom images
        extension (str): Extension of the image to save

    """

    """    for i in range(0, len(images_list), batch_size):
        print(images_list[i:i + batch_size])
        # more logic here"""
    # Create a dataframe that takes into account all the informations about the images, bounding boxes


    # Bounding Boxes
    dx_box = []
    sx_box = []
    all_box = []
    y_s = []
    imgs_id = []
    imgs_name = []

    for image_name in images_list:
        # Open an image and try to predict the segmentation_brixia mask and the alignment
        # The image is a 512x512x1 numpy array

        image_, original_size = convert_dicom_to_image_BRIXIA(path_to_data=path_to_data, image=image_name, transforms=transforms)
        img_i_preprocessed = preprocess(image_[:, :, 0], clip_limit=0.01, med_filt=3)[None, :, :, None]
        input_tensor = tf.convert_to_tensor(img_i_preprocessed, dtype=tf.float32)
        """"""


        #X, original_shape, imgs_name = loadData((512, 512), img_dir='data/AIforCOVID/imgs_r2')
        #image_ = np.array(Image.fromarray(np.array(X[2,:,:,0])))

        # TODO this part is necessary to show the capacity of the alignment network to rotate the image it can handle, -30 but not +30
        #  img_i_preprocessed = imutils.rotate(img_i_preprocessed, -30)

        """
        # -- NETWORK ALIGNMENT
        # 0) Predict the segmentation_brixia mask and the alignment
        img_i_preprocessed = preprocess(image_[:, :, 0], clip_limit=0.01, med_filt=3)
        img_i_preprocessed = imutils.rotate(img_i_preprocessed, -30)[None, :, :, None]
        
        segmentation_aligned, parameters_tensor = mask_aligner.predict(input_tensor)
        segmentation_aligned = segmentation_aligned > threshold
        props_aligned = measure.regionprops(segmentation_aligned.squeeze().astype(int))  # returns image properties

        segmentation_aligned = np.squeeze(segmentation_aligned)
        segmentation_aligned = segmentation_aligned > threshold  # Threshold the mask
        Apply the alignement to the original image
        input_shape = (512, 512, 1)
        BI = BilinearInterpolation(input_shape[:2])
        return_interpolated, parameters_tensor = BI.call((input_tensor, parameters_tensor))
        print(return_interpolated.shape) """

        # 1) SEGMENTATION: Predict the segmentation_brixia mask for the original mask
        segmentation = Segmenter.predict(input_tensor)
        segmentation = segmentation > threshold# Threshold the mask
        segmentation = np.array(segmentation).squeeze()
        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(segmentation, 0.02 * np.prod((512,512)))

        # plot
        # make the image smooth
        image_gauss = skimage.filters.gaussian(pr, sigma=9)
        # automatic threshold
        image_th = skimage.filters.threshold_otsu(image_gauss)
        # find contour

        mask = image_gauss > image_th

        mask_th = mask.astype(np.int32).astype(np.uint8)
        pr_ = pr.astype(np.int32).astype(np.uint8)

        contours, _ = cv2.findContours(mask_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        original_contours, _ = cv2.findContours(pr_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        empty, empty_original = None, None
        empty = np.zeros(mask.shape)[:,:,None]
        empty_original = np.zeros(mask.shape)[:,:,None]
        for i, (con, or_con) in enumerate(zip(contours, original_contours)):
            if i + 1 > 2:
                break
            else:
                empty = cv2.fillPoly(empty, [con], (255, 255, 255))
                empty_original= cv2.fillPoly(empty_original, [or_con], (255, 255, 255))
        # Thresholded AND Original
        Mask_OR = Volume_mask_and_or(empty, empty_original)
        final_mask = Mask_OR.astype(np.bool8)  # new mask
        # Updating mask

        # get box for single lungs
        lbl = measure.label(final_mask[:,:,0])
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
        # Get label
        y = list(metadata[[image_name in val for val in metadata['Filename'].to_list()]]['BrixiaScore'].array)
        y_s.append(y)

        # Save path

        image_extension = '.tiff'
        dest_path = os.path.join(os.path.dirname(path_to_data), 'processed')

        """# squeeze the image to remove the channel dimension
        image_aligned = np.squeeze(image_aligned)
        os.makedirs(dest_path, exist_ok=True)"""

        segmentation_folder = os.path.join(dest_path, 'masks')
        os.makedirs(segmentation_folder, exist_ok=True)
        """        images_aligned_folder = path_to_data.replace('images', 'images_aligned')
        os.makedirs(images_aligned_folder, exist_ok=True)
        segmentation_aligned_folder = path_to_data.replace('images', 'mask_aligned')
        os.makedirs(segmentation_aligned_folder, exist_ok=True)"""
        # Conversion to PIL
        final_mask = final_mask[:,:,0].astype(np.uint8) * 255
        final_mask = Image.fromarray(final_mask.astype(np.uint8))
        final_mask.resize(original_size)
        # Create the function to save the image
        file_dest = os.path.join(segmentation_folder, image_name.replace('.dcm', image_extension))
        img_saver = lambda x: x.save(file_dest)
        imgs_id.append(image_name.split('.')[0])
        imgs_name.append(file_dest)
        img_saver(final_mask)



        """img_saver(Image.fromarray(image_aligned), 'aligned_', images_aligned_folder)
        img_saver(Image.fromarray(segmentation_aligned), 'mask_aligned_', segmentation_aligned_folder)"""

    # Save all the informations about the dataset segmented
    # save excel with boxes
    bounding_box = pd.DataFrame(index=imgs_id)
    bounding_box["dx"] = dx_box
    bounding_box["sx"] = sx_box
    bounding_box["all"] = all_box
    bounding_box["label"] = y_s
    bounding_box["img_path"] = imgs_name
    # dropna
    bounding_box = bounding_box.dropna(subset=["label"])
    bounding_box_file = os.path.join(dest_path, 'box_data.xlsx')
    bounding_box.to_excel(bounding_box_file, index=True, index_label="img")


if __name__ == '__main__':

    # This code is needed to obtain segmentation_brixia mask and lungs alignment
    tf.config.get_visible_devices()
    Mask_Aligner, Segmenter, Parameters = AlignerSegmenter(
              backbone_name='resnet18',
              input_shape=(512, 512, 1),
              input_tensor=None,
              encoder_weights=None,
              freeze_encoder=True,
              skip_connections='default',
              decoder_block_type='transpose',
              decoder_filters=(256, 128, 64, 32, 16),
              decoder_use_batchnorm=True,
              n_upsample_blocks=5,
              upsample_rates=(2, 2, 2, 2, 2),
              activation='sigmoid',
              load_seg_model=True,
              seg_model_weights='weights/segmentation_brixia/segmentation_brixia-model.h5',
              freeze_segmentation=True,
              load_align_model=True,
              align_model_weights='weights/alignment/alignment-model.h5',
              freeze_align_model=True,
              pretrain_aligment_net=False,
              )

    print('Started preprossesing of BrixIA')

    dataset_selector = {0: 'BRIXIA', 1: 'AIforCOVID', 2: 'COHEN'}
    # Select the dataset to preprocess
    dataset = 0

    base_data_folder = f'data/{dataset_selector[dataset]}'
    path_to_data = os.path.join(base_data_folder ,'dicom_clean')

    processes = 1
    # Iterate over all the images inside the folder
    # Loading all the clean Dicom files

    images_list = os.listdir(path_to_data)
    metadata = 'data/BRIXIA/metadata_global_v2.csv'
    metadata = pd.read_csv(metadata, sep=';')
    # Resizer
    size = (512, 512)

    transforms = torch_transforms = torch_transforms.Compose([
        torch_transforms.Resize(size)
    ])


    # Call the function to create all the masks
    create_mask(Segmenter=Segmenter,
                mask_aligner=Mask_Aligner,
                lung_aligner=Parameters,
                path_to_data=path_to_data,
                metadata=metadata,
                alignment=False,
                images_list=images_list,
                dataset=dataset_selector[dataset],
                output_dir=os.path.join(base_data_folder, 'processed'),
                transforms=transforms
                )

    print('Finished preprocessing.')


















