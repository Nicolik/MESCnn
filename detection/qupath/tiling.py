import os
import cv2
import shutil
from detection.qupath.utils import tile_region, is_foreground, is_black


def dir_name_from_wsi(path):
    collate = 2 if ('.ome.tif' in path or '.ome.tiff' in path) else 1
    return '.'.join(path.split('.')[:-collate])


class BaseTiler(object):
    def __init__(self, reader, out_dir, tile_ext='jpeg'):
        self.reader = reader
        out_dir = os.path.join(out_dir, dir_name_from_wsi(reader.name))
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.tile_ext = tile_ext

    def tile_image(self, desired_op, tile_size, tile_stride):
        pass


class WholeTilerOpenslide(BaseTiler):
    def __init__(self, reader, out_dir, tile_ext='jpeg'):
        super().__init__(reader, out_dir, tile_ext=tile_ext)

    def tile_image(self, desired_op, tile_size, tile_stride, check_fg=True):
        roi = (0, 0, *self.reader.dimensions)
        tile_coords = tile_region(roi, tile_size, tile_stride)
        for tile_coord in tile_coords:
            x, y, h, w = tile_coord
            tile_image = self.reader.read_resolution(x, y, h, w, desired_op, do_rescale=True, read_bgr=True)
            tile_name = f"{self.reader.name}__OP_{desired_op}__ROI_{x}_{y}_{h}_{w}.{self.tile_ext}"
            if not is_black(tile_image):
                if not check_fg or is_foreground(tile_image):
                    tile_path = os.path.join(self.out_dir, tile_name)
                    print(f"Writing to {tile_path}")
                    cv2.imwrite(tile_path, tile_image)


class WholeTilerBioformats(BaseTiler):
    def __init__(self, reader, out_dir, tile_ext='jpeg'):
        super().__init__(reader, out_dir, tile_ext=tile_ext)

    def tile_image(self, desired_op, tile_size, tile_stride, check_fg=True):
        for i, dimensions in enumerate(self.reader.dimensions):
            s = self.reader.indexes[i]
            roi = (0, 0, *dimensions)
            tile_coords = tile_region(roi, tile_size, tile_stride)
            for tile_coord in tile_coords:
                x, y, h, w = tile_coord
                tile_image = self.reader.read_resolution(s, x, y, h, w, desired_op, do_rescale=True, read_bgr=True)
                tile_name = f"{self.reader.name}_{s}__OP_{desired_op}__ROI_{x}_{y}_{h}_{w}.{self.tile_ext}"
                if tile_image is not None:
                    if not is_black(tile_image):
                        if not check_fg or is_foreground(tile_image):
                            tile_path = os.path.join(self.out_dir, tile_name)
                            print(f"Writing to {tile_path}")
                            cv2.imwrite(tile_path, tile_image)
