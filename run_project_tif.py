import os
import shutil
import logging
import subprocess
from definitions import ROOT_DIR
from mescnn.classification.gutils.config import OxfordModelNameCNN
from mescnn.detection.model.config import SegmentationModelName
from mescnn.detection.qupath.config import PathMESCnn, PathWSI, get_test_wsis
from mescnn.detection.qupath.download import download_project, download_slide, sanitize_qupath_project

# Tests
download_qp = True
test_segment_project = True
test_tile = True
test_segment = True
test_qu2json = True
test_json2exp = True
test_classify = True

qupath_empty_dir = PathWSI.QUPATH_MESCnn_DIR_NOANN
detection_model = SegmentationModelName.CASCADE_R_50_FPN_1x
path_to_export = os.path.join(PathWSI.MESCnn_EXPORT, detection_model)
qupath_segm_dir = os.path.join(path_to_export, 'QuPathProject')
wsi_tiff_dir = PathWSI.MESCnn_WSI

if download_qp:
    if not os.path.exists(qupath_empty_dir):
        download_project(PathWSI.MESCnn_DATASET)
    for wsi in get_test_wsis():
        wsi_id = os.path.basename(wsi)
        path_to_wsi_file = os.path.join(wsi_tiff_dir, wsi_id)
        if not os.path.exists(path_to_wsi_file):
            logging.info(f"Downloading {wsi_id}...")
            download_slide(wsi_id, os.path.dirname(wsi_tiff_dir))
    sanitize_qupath_project(qupath_empty_dir)

logging.info(f"QuPathProject: {qupath_segm_dir}")
if test_segment_project:
    if os.path.exists(qupath_segm_dir):
        shutil.rmtree(qupath_segm_dir)
    shutil.copytree(qupath_empty_dir, qupath_segm_dir)

    logging.info(f"Running {PathMESCnn.SEGMENT_PROJECT}")
    subprocess.run(["python", PathMESCnn.SEGMENT_PROJECT,
                    "--wsi", wsi_tiff_dir,
                    "--export", path_to_export,
                    "--qupath", qupath_segm_dir,
                    "--model", detection_model,
                    "--do-segment", str(test_segment),
                    "--do-tiling", str(test_tile)])
else:
    logging.info(f"Skipping run of {PathMESCnn.SEGMENT_PROJECT}")

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
