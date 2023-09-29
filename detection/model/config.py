from detectron2.config import get_cfg
from detectron2 import model_zoo


class DetectronConfigFile:
    R_50_DC5_1x = "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml"
    R_50_DC5_3x = "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"
    R_50_C4_3x = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
    R_50_FPN_3x = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    R_101_FPN_3x = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    R_50_C4_1x = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
    CASCADE_R_50_FPN_3x = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    CASCADE_R_50_FPN_1x = "Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml"


class DetectronModelName:
    R_50_DC5_1x = "mask_rcnn_R_50_DC5_1x"
    R_50_DC5_3x = "mask_rcnn_R_50_DC5_3x"
    R_50_C4_3x = "mask_rcnn_R_50_C4_3x"
    R_50_FPN_3x = "mask_rcnn_R_50_FPN_3x"
    R_101_FPN_3x = "mask_rcnn_R_101_FPN_3x"
    R_50_C4_1x = "mask_rcnn_R_50_C4_1x"
    CASCADE_R_50_FPN_3x = "cascade_mask_rcnn_R_50_FPN_3x"
    CASCADE_R_50_FPN_1x = "cascade_mask_rcnn_R_50_FPN_1x"


class SegmentationModelName:
    R_50_DC5_1x = "R_50_DC5_1x"
    R_50_DC5_3x = "R_50_DC5_3x"
    R_50_C4_3x = "R_50_C4_3x"
    R_50_FPN_3x = "R_50_FPN_3x"
    R_101_FPN_3x = "R_101_FPN_3x"
    R_50_C4_1x = "R_50_C4_1x"
    CASCADE_R_50_FPN_3x = "cascade_R_50_FPN_3x"
    CASCADE_R_50_FPN_1x = "cascade_R_50_FPN_1x"


PAPER_SEGMENTATION_ARCHITECTURES = [
    SegmentationModelName.R_50_C4_1x,
    SegmentationModelName.R_50_C4_3x,
    SegmentationModelName.R_50_FPN_3x,
    SegmentationModelName.R_101_FPN_3x,
    SegmentationModelName.R_50_DC5_1x,
    SegmentationModelName.R_50_DC5_3x,
    SegmentationModelName.CASCADE_R_50_FPN_1x,
    SegmentationModelName.CASCADE_R_50_FPN_3x
]


DEFAULT_SEGMENTATION_MODEL = SegmentationModelName.CASCADE_R_50_FPN_1x
CLI_MODEL_NAME_DICT = {
    SegmentationModelName.R_50_DC5_1x: (DetectronConfigFile.R_50_DC5_1x, DetectronModelName.R_50_DC5_1x),
    SegmentationModelName.R_50_DC5_3x: (DetectronConfigFile.R_50_DC5_3x, DetectronModelName.R_50_DC5_3x),
    SegmentationModelName.R_50_C4_3x: (DetectronConfigFile.R_50_C4_3x, DetectronModelName.R_50_C4_3x),
    SegmentationModelName.R_50_FPN_3x: (DetectronConfigFile.R_50_FPN_3x, DetectronModelName.R_50_FPN_3x),
    SegmentationModelName.R_101_FPN_3x: (DetectronConfigFile.R_101_FPN_3x, DetectronModelName.R_101_FPN_3x),
    SegmentationModelName.R_50_C4_1x: (DetectronConfigFile.R_50_C4_1x, DetectronModelName.R_50_C4_1x),
    SegmentationModelName.CASCADE_R_50_FPN_1x: (DetectronConfigFile.CASCADE_R_50_FPN_1x, DetectronModelName.CASCADE_R_50_FPN_1x),
    SegmentationModelName.CASCADE_R_50_FPN_3x: (DetectronConfigFile.CASCADE_R_50_FPN_3x, DetectronModelName.CASCADE_R_50_FPN_3x)
}


def get_detection_models():
    return list(CLI_MODEL_NAME_DICT.keys())


class DetectronDataset:
    TRAIN_GLOMERULI = "Glomeruli_Detection_Train"
    TEST_GLOMERULI = "Glomeruli_Detection_Test"
    EXT_VAL_GLOMERULI = "Glomeruli_Detection_ExtVal"


class Things:
    GLOMERULUS = "glomerulus"


dataset_subsets = {"train", "test", "extval"}


def build_model_config(config_file):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0003
    cfg.SOLVER.MAX_ITER = 300000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    return cfg


def config_dataset_internal_val(cfg):
    cfg.DATASETS.TRAIN = [DetectronDataset.TRAIN_GLOMERULI]
    cfg.DATASETS.TEST = [DetectronDataset.TEST_GLOMERULI,
                         DetectronDataset.EXT_VAL_GLOMERULI]


def config_dataset_external_val(cfg):
    cfg.DATASETS.TRAIN = [DetectronDataset.TRAIN_GLOMERULI,
                          DetectronDataset.TEST_GLOMERULI]
    cfg.DATASETS.TEST = [DetectronDataset.EXT_VAL_GLOMERULI]


def config_dataset_all_training(cfg):
    cfg.DATASETS.TRAIN = [DetectronDataset.TRAIN_GLOMERULI,
                          DetectronDataset.TEST_GLOMERULI,
                          DetectronDataset.EXT_VAL_GLOMERULI]
    cfg.DATASETS.TEST = []


def set_config(cfg, train_config):
    if train_config in ["i", "internal"]:
        config_dir = "internal-validation"
        config_dataset_internal_val(cfg)
    elif train_config in ["e", "external"]:
        config_dir = "external-validation"
        config_dataset_external_val(cfg)
    elif train_config in ["a", "all"]:
        config_dir = "all-training"
        config_dataset_all_training(cfg)
    return config_dir
