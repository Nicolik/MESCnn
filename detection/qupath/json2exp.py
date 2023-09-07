import json
import os
import logging
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw

import javabridge
import bioformats

from classification.gutils.image import apply_mask_crop
from detection.io.bioformats_reader import BioformatsReader
from detection.io.openslide_reader import OpenslideReader
from detection.qupath.paths import get_reader_type
from detection.qupath.utils import ReaderType


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export Annotations from JSON to Image Files')
    parser.add_argument('-e', '--export', type=str, help='path/to/export', required=True)
    parser.add_argument('-w', '--wsi-dir', type=str, help='path/to/wsi/dir', required=True)
    args = parser.parse_args()

    javabridge.start_vm(class_path=bioformats.JARS)

    path_to_export = args.export
    path_to_wsi = args.wsi_dir

    path_crops = os.path.join(path_to_export, "Temp", "qu2json-output", "rois.csv")
    path_dataset = os.path.join(path_to_export, "Temp", "qu2json-output", "dataset_detectron2.json")

    df_crops = pd.read_csv(path_crops)
    with open(path_dataset, 'r') as fp:
        dataset_dict = json.load(fp)

    # dataset_dict: List[Dict<file_name, height, width, image_id, annotations>]
    # annotations: List[Dict<bbox, bbox_mode, category_id, segmentation>]

    assert len(dataset_dict) == len(df_crops), "Mismatch between crops and annotations!"

    path_to_export_json2exp = os.path.join(path_to_export, "Temp", "json2exp-output")
    os.makedirs(path_to_export_json2exp, exist_ok=True)

    path_to_original = os.path.join(path_to_export_json2exp, "Original")
    path_to_mask = os.path.join(path_to_export_json2exp, "Mask")
    path_to_crop = os.path.join(path_to_export_json2exp, "Crop")
    path_to_crop_256 = os.path.join(path_to_export_json2exp, "Crop-256")

    os.makedirs(path_to_original, exist_ok=True)
    os.makedirs(path_to_mask, exist_ok=True)
    os.makedirs(path_to_crop, exist_ok=True)
    os.makedirs(path_to_crop_256, exist_ok=True)

    for idx, row in df_crops.iterrows():
        print(f"Iter: {(idx+1):4d} / {len(df_crops):4d}")
        ext = row['ext']
        reader_type = get_reader_type(ext)
        print(f"Ext: {ext}, Reader Type: {reader_type}")

        path_to_wsi = row['path-to-wsi']
        x, y = row['x'], row['y']
        xsize, ysize = row['w'], row['h']
        idx_s = row['s']

        if reader_type in [ReaderType.SCN, ReaderType.OME_TIFF]:
            image_os = BioformatsReader(path_to_wsi)
            orig = image_os.read_resolution(image_os.indexes[idx_s], x, y, xsize, ysize)
        elif reader_type in [ReaderType.NDPI, ReaderType.SVS, ReaderType.MRXS, ReaderType.TIFF]:
            image_os = OpenslideReader(path_to_wsi)
            orig = image_os.read_region((x, y), 0, (xsize, ysize))
        else:
            logging.error(f"[json2exp] reader_type '{reader_type}' invalid for wsi '{path_to_wsi}'!")
            continue

        if orig is not None:
            if orig.shape[0] != ysize or orig.shape[1] != xsize:
                logging.error(f"[json2exp] Tile [{x}, {y}, {xsize}, {ysize}] from '{path_to_wsi}' has a shape of {orig.shape}!")
                continue
        else:
            logging.error(f"[json2exp] Tile [{x}, {y}, {xsize}, {ysize}] from '{path_to_wsi}' returns None!")
            continue

        wsi_id = row['image-id']
        subdir_original = os.path.join(path_to_original, wsi_id)
        os.makedirs(subdir_original, exist_ok=True)
        subdir_mask = os.path.join(path_to_mask, wsi_id)
        os.makedirs(subdir_mask, exist_ok=True)
        subdir_crop = os.path.join(path_to_crop, wsi_id)
        os.makedirs(subdir_crop, exist_ok=True)
        subdir_crop_256 = os.path.join(path_to_crop_256, wsi_id)
        os.makedirs(subdir_crop_256, exist_ok=True)

        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        orig_file = os.path.join(subdir_original, row['filename'])
        cv2.imwrite(orig_file, orig)

        crop_y = orig.shape[0]
        crop_x = orig.shape[1]
        rescale_factor_y = ysize / crop_y
        rescale_factor_x = xsize / crop_x

        mask_file = os.path.join(subdir_mask, row['filename'])
        mask_image = Image.new('L', (crop_x, crop_y), 0)
        annotations = dataset_dict[idx]['annotations']
        for annotation in annotations:
            polygon = annotation['segmentation']
            ImageDraw.Draw(mask_image).polygon(polygon[0], outline=1, fill=1)
        mask = np.array(mask_image)
        cv2.imwrite(mask_file, mask)

        cmask_file = os.path.join(subdir_crop, row['filename'])
        masked_image = apply_mask_crop(orig, mask, only_largest=True, smooth_mask=True)
        cv2.imwrite(cmask_file, masked_image)

        cmask256_file = os.path.join(subdir_crop_256, row['filename'])
        masked_image_res = cv2.resize(masked_image, (256, 256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(cmask256_file, masked_image_res)

    javabridge.kill_vm()
