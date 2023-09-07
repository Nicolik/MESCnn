import os
from enum import IntEnum

from definitions import ROOT_DIR

MAGNIFICATION = 40
MIN_AREA_GLOMERULUS_UM = 5000
MIN_AREA_BBOX_GLOMERULUS = 5000
DETECTRON_SCORE_THRESHOLD = 0.5


class PathMESCnn:
    SEGMENT_PROJECT = os.path.join(ROOT_DIR, "detection", "qupath", "segment_project.py")
    TILE = os.path.join(ROOT_DIR, "detection", "qupath", "tile.py")
    SEGMENT = os.path.join(ROOT_DIR, "detection", "qupath", "segment.py")
    PKL2QU = os.path.join(ROOT_DIR, "detection", "qupath", "pkl2qu.py")
    QU2JSON = os.path.join(ROOT_DIR, "detection", "qupath", "qu2json.py")
    JSON2EXP = os.path.join(ROOT_DIR, "detection", "qupath", "json2exp.py")
    INFERENCE = os.path.join(ROOT_DIR, "detection", "model", "inference.py")
    CLASSIFY = os.path.join(ROOT_DIR, "classification", "inference", "mesc", "classify.py")
    COLLATE_CLASSIFY = os.path.join(ROOT_DIR, "classification", "inference", "mesc", "collate_classify.py")


class PathWSI:
    BASE_MESCnn = os.path.join(ROOT_DIR, 'Data')
    MESCnn_EXPORT = os.path.join(BASE_MESCnn, 'Export')

    MESCnn_DATASET = os.path.join(BASE_MESCnn, 'Dataset')
    QUPATH_MESCnn_DIR_NOANN = os.path.join(MESCnn_DATASET, 'QuPathProject-NoAnnotations')

    MESCnn_WSI = os.path.join(MESCnn_DATASET, 'WSI')
    MESCnn_WSI_BARI = os.path.join(MESCnn_WSI, 'bari_sample_slide.ome.tif')
    MESCnn_WSI_COLOGNE = os.path.join(MESCnn_WSI, 'cologne_sample_slide.ome.tif')
    MESCnn_WSI_COLOGNE_2 = os.path.join(MESCnn_WSI, 'cologne_sample_slide_2.ome.tif')
    MESCnn_WSI_SZEGED = os.path.join(MESCnn_WSI, 'szeged_sample_slide.ome.tif')


def get_test_wsis():
    return [
        PathWSI.MESCnn_WSI_BARI,
        PathWSI.MESCnn_WSI_COLOGNE,
        PathWSI.MESCnn_WSI_COLOGNE_2,
        PathWSI.MESCnn_WSI_SZEGED
    ]


class GlomerulusDetection(IntEnum):
    BACKGROUND = 0
    GLOMERULUS = 1


def init_data_dict():
    return {
        'image-id': [],
        'filename': [],
        'path-to-wsi': [],
        'ext': [],
        's': [],
        'x': [],
        'y': [],
        'w': [],
        'h': []
    }
