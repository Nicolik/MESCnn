import os
import subprocess
import logging

from classification.gutils.utils import str2bool
from detection.model.config import DEFAULT_SEGMENTATION_MODEL
from detection.qupath.config import PathMESCnn, MAGNIFICATION
from detection.qupath.download import download_slide
from detection.qupath.utils import qupath2list

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segment Glomeruli from QuPathProject with Detectron2')
    parser.add_argument('-w', '--wsi', type=str, help='path/to/wsi/dir', required=True)
    parser.add_argument('-e', '--export', type=str, help='path/to/export', required=True)
    parser.add_argument('-q', '--qupath', type=str, help='path/to/qupath', required=True)

    parser.add_argument('--do-download', type=str2bool, help='True for downloading slides; False otherwise', default=False)
    parser.add_argument('--do-segment', type=str2bool, help='True for segmentation; False otherwise', default=True)
    parser.add_argument('-m', '--model', type=str, help='Model to use for inference', default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument('-c', '--train-config', type=str, help='I=Internal/E=External/A=All', default="external")

    parser.add_argument('--do-tiling', type=str2bool, help='True for tiling; False otherwise', default=True)
    parser.add_argument('--desired-op', type=int, help='Desired Magnification for performing segmentation', default=10)
    parser.add_argument('--tile-size', type=int, nargs=2, help='Size of a tile (in pixel, at maximum magnification available)', default=(4096, 4096))
    parser.add_argument('--tile-stride', type=int, nargs=2, help='Size of the stride between tiles (in pixel, at maximum magnification available)', default=(2048, 2048))

    args = parser.parse_args()
    wsi_list = qupath2list(args.qupath)

    for wsi in wsi_list:
        print(f"[segment_project] wsi: {wsi}")
        path_to_wsi_file = os.path.join(args.wsi, wsi)

        if args.do_download:
            if not os.path.exists(path_to_wsi_file):
                logging.info(f"Downloading {wsi}...")
                download_slide(wsi, os.path.dirname(args.wsi))

        if args.do_tiling:
            logging.info(f"{PathMESCnn.TILE} running on {path_to_wsi_file}...")
            subprocess.run(["python", PathMESCnn.TILE,
                            "--wsi", path_to_wsi_file,
                            "--export", args.export,
                            "--desired-op", str(args.desired_op),
                            "--tile-size", f"{args.tile_size[0]}", f"{args.tile_size[1]}",
                            "--tile-stride", f"{args.tile_stride[0]}", f"{args.tile_stride[1]}"])

        if args.do_segment:
            logging.info(f"{PathMESCnn.SEGMENT} running on {path_to_wsi_file}...")
            undersampling = MAGNIFICATION // args.desired_op
            subprocess.run(["python", PathMESCnn.SEGMENT,
                            "--wsi", path_to_wsi_file,
                            "--export", args.export,
                            "--qupath", args.qupath,
                            "--model", args.model,
                            "--train-config", args.train_config,
                            "--undersampling", str(undersampling)])
