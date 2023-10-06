import os
import logging
import subprocess
import shutil

from definitions import ROOT_DIR
from mescnn.classification.gutils.config import OxfordModelNameCNN
from mescnn.detection.model.config import get_detection_models
from mescnn.detection.qupath.config import PathMESCnn, PathWSI, get_test_wsis
from mescnn.detection.qupath.download import download_slide

wsis = get_test_wsis()
detection_models = get_detection_models()

# Tests
download_slides = True
test_tile = True
test_segment = True
test_qu2json = True
test_json2exp = True
test_classify = True

wsi_tiff_dir = PathWSI.MESCnn_WSI
path_to_export_base = os.path.join(PathWSI.MESCnn_EXPORT)
train_config = "all"

if download_slides:
    for wsi in wsis:
        if not os.path.exists(wsi):
            print(f"Downloading {wsi}...")
            slide_name = os.path.basename(wsi)
            slide_path = download_slide(slide_name, PathWSI.MESCnn_DATASET)
            print(f"Downloaded: {slide_path}!")

for detection_model in detection_models:

    path_to_export = os.path.join(PathWSI.MESCnn_EXPORT, detection_model)
    qupath_segm_dir = os.path.join(path_to_export, 'QuPathProject')

    if test_tile:
        for wsi in wsis:
            logging.info(f"{PathMESCnn.TILE} running on {wsi}...")
            subprocess.run(["python", PathMESCnn.TILE,
                            "--wsi", wsi,
                            "--export", path_to_export])

    if test_segment:
        for wsi in wsis:
            logging.info(f"{PathMESCnn.SEGMENT} running on {wsi}...")
            subprocess.run(["python", PathMESCnn.SEGMENT,
                            "--wsi", wsi,
                            "--export", path_to_export,
                            "--qupath", qupath_segm_dir,
                            "--model", detection_model,
                            "--train-config", train_config])
    else:
        logging.info(f"Skipping run of {PathMESCnn.SEGMENT}!")

    if test_qu2json:
        logging.info(f"Running {PathMESCnn.QU2JSON}")
        subprocess.run(["python", PathMESCnn.QU2JSON,
                        "--export", path_to_export,
                        "--wsi-dir", wsi_tiff_dir,
                        "--qupath", qupath_segm_dir])
    else:
        logging.info(f"Skipping run of {PathMESCnn.QU2JSON}")

    if test_json2exp:
        logging.info(f"Running {PathMESCnn.JSON2EXP}")
        subprocess.run(["python", PathMESCnn.JSON2EXP,
                        "--export", path_to_export,
                        "--wsi-dir", wsi_tiff_dir])
    else:
        logging.info(f"Skipping run of {PathMESCnn.JSON2EXP}")

    if test_classify:
        net_M = OxfordModelNameCNN.EfficientNet_V2_M
        net_E = OxfordModelNameCNN.EfficientNet_V2_M
        net_S = OxfordModelNameCNN.DenseNet161
        net_C = OxfordModelNameCNN.MobileNet_V2
        use_vit = False

        use_vit_M = use_vit_E = use_vit_S = use_vit_C = use_vit
        logging.info(f"Running {PathMESCnn.CLASSIFY} with {net_M}, {net_E}, {net_S}, {net_C}")
        subprocess.run(["python", PathMESCnn.CLASSIFY,
                        "--root-path", ROOT_DIR,
                        "--export-dir", path_to_export,
                        "--netM", net_M, "--vitM", str(use_vit_M),
                        "--netE", net_E, "--vitE", str(use_vit_E),
                        "--netS", net_S, "--vitS", str(use_vit_S),
                        "--netC", net_C, "--vitC", str(use_vit_C)])
    else:
        logging.info(f"Skipping run of {PathMESCnn.CLASSIFY}")
