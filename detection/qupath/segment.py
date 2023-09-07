import logging
import os
import tqdm
import cv2
import numpy as np
import time
import pickle
import subprocess

from detectron2.engine import DefaultPredictor

print("Loading local libraries...")
from definitions import ROOT_DIR
from detection.model.config import build_model_config, CLI_MODEL_NAME_DICT, set_config, DEFAULT_SEGMENTATION_MODEL
from detection.qupath.config import MIN_AREA_GLOMERULUS_UM, DETECTRON_SCORE_THRESHOLD, PathMESCnn
from detection.qupath.utils import get_dataset_dicts_validation, tile2xywh, mask2polygon, get_area_10x
from detection.qupath.nms import nms
from detection.qupath.tiling import dir_name_from_wsi
from detection.qupath.download import download_detector
print("Local libraries loaded!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segment Glomeruli with Detectron2 from WSI')
    parser.add_argument('-w', '--wsi', type=str, help='path/to/wsi', required=True)
    parser.add_argument('-e', '--export', type=str, help='path/to/export', required=True)
    parser.add_argument('-q', '--qupath', type=str, help='path/to/qupath', required=True)
    parser.add_argument('-m', '--model', type=str, help='Model to use for inference', default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument('-c', '--train-config', type=str, help='I=Internal/E=External/A=All', default="external")
    parser.add_argument('--undersampling', type=int, help='Undersampling factor of tiles', default=4)

    args = parser.parse_args()
    if args.model not in CLI_MODEL_NAME_DICT:
        logging.warning(f"Model '{args.model}' not present, default to {DEFAULT_SEGMENTATION_MODEL}!")
        args.model = DEFAULT_SEGMENTATION_MODEL

    train_config = args.train_config
    config_file, model_name = CLI_MODEL_NAME_DICT[args.model]

    cfg = build_model_config(config_file)
    config_dir = set_config(cfg, train_config)

    undersampling = args.undersampling

    model_folder = os.path.join(ROOT_DIR, 'detection', 'logs', model_name, config_dir)
    logs_dir = os.path.join(model_folder, 'output')
    path_to_weights = os.path.join(logs_dir, "model_final.pth")
    if not os.path.exists(path_to_weights):
        print(f"Model weights not found: {path_to_weights}!")
        model_path = download_detector(model_name, config_dir)
        print(f"Downloaded: {model_path}")
    else:
        print(f"Model weights found: {path_to_weights}!")

    cfg.MODEL.WEIGHTS = path_to_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTRON_SCORE_THRESHOLD
    predictor = DefaultPredictor(cfg)

    path_to_wsi = args.wsi
    wsi_name = dir_name_from_wsi(os.path.basename(path_to_wsi))
    path_to_qigs_qupath = args.qupath
    tile_dir = os.path.join(args.export, 'Temp', 'tiler-output', 'Tiles', wsi_name)
    path_to_segment_output = os.path.join(args.export, 'Temp', 'segment-output')

    mask_dir = os.path.join(path_to_segment_output, 'Masks', wsi_name)
    mask_picked_dir = os.path.join(path_to_segment_output, 'Masks-Picked', wsi_name)
    detection_dir = os.path.join(path_to_segment_output, 'Detections', wsi_name)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mask_picked_dir, exist_ok=True)
    os.makedirs(detection_dir, exist_ok=True)

    print(f"Attempting to build dataset dict from {tile_dir}")
    dataset_dicts = get_dataset_dicts_validation(tile_dir)

    masks_wsi = []
    bboxes_wsi = []
    scores_wsi = []
    offset_wsi = []

    counts = 0
    for dd, d in enumerate(tqdm.tqdm(dataset_dicts)):
        filename = d["file_name"]
        base_name = os.path.basename(filename)
        x1_off, y1_off, _, _ = tile2xywh(filename)

        logging.info(f"Basename: {base_name}, x_off: {x1_off}, y_off: {y1_off}")

        im = cv2.imread(filename)
        start_time = time.time()
        outputs = predictor(im)
        classes = outputs["instances"].get("pred_classes").cpu().numpy()
        scores = outputs["instances"].get("scores").cpu().numpy()

        mask_array = outputs['instances'].to("cpu").pred_masks.numpy()
        mask_array = mask_array.astype(np.uint8)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"[Detectron2] Elapsed Time (sec): {elapsed_time:.2f}")

        for m, mask in enumerate(mask_array):
            logging.info(f"Mask ({m}) - shape: {mask.shape}, dtype: {mask.dtype}, sum: {mask.sum()}")
            path_to_mask = os.path.join(mask_dir, base_name + f"_{m}.png")
            logging.info(f"Writing Mask to {path_to_mask}")
            cv2.imwrite(path_to_mask, mask*255)

            contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

            if bounding_boxes:
                x1, y1 = bounding_boxes[0][0]*undersampling + int(x1_off), bounding_boxes[0][1]*undersampling + int(y1_off)
                w, h = bounding_boxes[0][2]*undersampling, bounding_boxes[0][3]*undersampling
                x2, y2 = x1 + w, y1 + h
                bboxes_wsi.append((x1, y1, x2, y2))
                scores_wsi.append(scores[m])
                masks_wsi.append(path_to_mask)
                offset_wsi.append((x1_off, y1_off))
                counts = counts + 1

    print(f"Before NMS: {len(bboxes_wsi)}")
    idxs = nms(bboxes_wsi, scores_wsi, threshold_iou=0.4, threshold_iom=0.4, return_idxs=True)
    print(f"After  NMS: {len(idxs)}")

    picked_boxes = [bboxes_wsi[i] for i in idxs]
    picked_score = [scores_wsi[i] for i in idxs]
    picked_masks = [masks_wsi[i] for i in idxs]
    picked_offset = [offset_wsi[i] for i in idxs]

    list_polygons = []
    glomerular_areas = []

    for glomerulus_roi, path_glomerulus_mask, xy_offset in zip(picked_boxes, picked_masks, picked_offset):
        x1_roi, y1_roi, x2_roi, y2_roi = glomerulus_roi
        w_roi, h_roi = x2_roi - x1_roi, y2_roi - y1_roi

        single_glomerulus_mask = cv2.imread(path_glomerulus_mask, cv2.IMREAD_GRAYSCALE)
        single_glomerulus_mask[single_glomerulus_mask>0] = 255
        single_glomerulus_mask = single_glomerulus_mask.astype(np.uint8)

        path_to_mask = os.path.join(mask_picked_dir, f"Glomerulus_ROI_{glomerulus_roi}.png")
        cv2.imwrite(path_to_mask, single_glomerulus_mask)

        polygon = mask2polygon(single_glomerulus_mask)
        area_um = get_area_10x(polygon)
        glomerular_areas.append(area_um)

        polygon_large = np.array([[point[0]*undersampling + xy_offset[0],
                                   point[1]*undersampling + xy_offset[1]] for point in polygon])

        if area_um > MIN_AREA_GLOMERULUS_UM:
            list_polygons.append(polygon_large)
        else:
            logging.warning(f"Area: {area_um} below min area of {MIN_AREA_GLOMERULUS_UM}!")

    logging.info(f"BBoxes before NMS: {len(bboxes_wsi)} / after NMS: {len(picked_boxes)}")

    ddict = {
        'Glomerulus': list_polygons
    }

    path_to_pickle = os.path.join(detection_dir, 'detections.pkl')
    with open(path_to_pickle, 'wb') as fp:
        pickle.dump(ddict, fp)

    subprocess.run(["python", PathMESCnn.PKL2QU,
                    "--wsi", path_to_wsi,
                    "--pickle", path_to_pickle,
                    "--qupath", path_to_qigs_qupath])
