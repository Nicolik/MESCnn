import logging
import os
import javabridge
import bioformats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract tiles from WSI')
    parser.add_argument('-w', '--wsi', type=str, help='path/to/wsi', required=True)
    parser.add_argument('-e', '--export', type=str, help='path/to/export', required=True)
    parser.add_argument('--desired-op', type=int, help='Desired Magnification for performing segmentation', default=10)
    parser.add_argument('--tile-size', type=int, nargs=2, help='Size of a tile (in pixel, at maximum magnification available)', default=(4096, 4096))
    parser.add_argument('--tile-stride', type=int, nargs=2, help='Size of the stride between tiles (in pixel, at maximum magnification available)', default=(2048, 2048))

    args = parser.parse_args()

    print("Starting JVM...")
    javabridge.start_vm(class_path=bioformats.JARS)
    print("JVM started!")

    print("Loading local libraries...")
    from detection.io.bioformats_reader import BioformatsReader
    from detection.io.openslide_reader import OpenslideReader
    from detection.qupath.tiling import WholeTilerOpenslide, WholeTilerBioformats
    print("Local libraries loaded!")

    path_to_wsi = args.wsi
    path_to_tiled = os.path.join(args.export, 'Temp', 'tiler-output', 'Tiles')

    if path_to_wsi.endswith(('.scn', '.ome.tiff', '.ome.tif')):
        reader = BioformatsReader(path_to_wsi)
        tiler = WholeTilerBioformats(reader, path_to_tiled)
    elif path_to_wsi.endswith(('.ndpi', '.svs', '.mrxs', '.tif', '.tiff')):
        reader = OpenslideReader(path_to_wsi)
        tiler = WholeTilerOpenslide(reader, path_to_tiled)

    desired_op = args.desired_op
    tile_size = args.tile_size
    tile_stride = args.tile_stride

    tiler.tile_image(desired_op, tile_size, tile_stride, check_fg=False)
    del reader
    javabridge.kill_vm()

    tile_dir = tiler.out_dir
    logging.info(f"The tiler has generated {len(os.listdir(tile_dir))} output tiles")
