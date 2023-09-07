import os
from enum import Enum

import cv2
import numpy as np


def tile_region(roi, tile_size, tile_stride):
    x, y, w, h = roi
    size_x, size_y = tile_size
    stride_x, stride_y = tile_stride

    tiles = []
    for xx in range(x, x+w, stride_x):
        for yy in range(y, y+h, stride_y):
            tiles.append([xx, yy, size_x, size_y])
    return tiles


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def PIL2cv2(PIL_image):
    """ Convert PIL image to cv2 image

    :param PIL_image: original PIL image
    :return: cv2 image
    """
    PIL_image = PIL_image.convert('RGB')
    opencv_img = np.array(PIL_image)
    return opencv_img


def is_foreground(image_np_rgb, THRESHOLD_BINARY=100, THRESHOLD_FOREGROUND=0.3):
    image_np_gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
    ret, thresh_np = cv2.threshold(image_np_gray, THRESHOLD_BINARY, 255, cv2.THRESH_BINARY_INV)
    image_size = image_np_gray.size
    foreground_pixels = thresh_np.sum() // 255
    ratio_foreground = foreground_pixels / image_size
    return ratio_foreground > THRESHOLD_FOREGROUND


def is_black(image_np_rgb, THRESHOLD_BLACK=5):
    image_np_gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
    return np.mean(image_np_gray < THRESHOLD_BLACK) > .1


class ReaderType(Enum):
    NONE = 0
    SVS  = 1
    NDPI = 2
    SCN  = 3
    MRXS = 4
    TIFF = 5
    OME_TIFF = 6


def get_dataset_dicts_validation(basepath):
    dataset_dicts = []
    filenames = os.listdir(basepath)

    for image in filenames:
        slide_id = image.split(".")[-1]
        record = {}

        filename = os.path.join(basepath, image)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = slide_id
        record["height"] = height
        record["width"] = width

        dataset_dicts.append(record)

    return dataset_dicts


def tile2xywh(filename):
    x_tile = int(filename.split('_')[-4])
    y_tile = int(filename.split('_')[-3])
    w_tile = int(filename.split('_')[-2])
    h_tile = int(filename.split('_')[-1].split('.')[0])
    return x_tile, y_tile, w_tile, h_tile


def mask2polygon(mask):
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = max(contours, key=cv2.contourArea)
    polygon = np.squeeze(polygon, axis=1)
    return polygon


def get_area_10x(polygon):
    polygon_small = np.array([[point[0], point[1]] for point in polygon])
    area_um = cv2.contourArea(polygon_small)
    return area_um


def qupath2list(path_to_qupath):
    from paquo.projects import QuPathProject
    wsi_list = []

    with QuPathProject(path_to_qupath, mode='r') as qp:
        num_images = len(qp.images)
        print(f"opened project '{qp.name}' with {num_images} images")
        for i, image in enumerate(qp.images):
            if '.scn' in image.image_name:
                if 'Series 1' in image.image_name:
                    wsi_list.append(str(image.image_name).split('.scn')[0] + '.scn')
            else:
                wsi_list.append(image.image_name)

    return wsi_list
