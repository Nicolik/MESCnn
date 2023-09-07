import os

from detection.qupath.utils import ReaderType


def get_mrxs_dir(image_name):
    id_mrxs = image_name.split('_')[0]
    pref = id_mrxs.split('-')[0]
    suff = id_mrxs.split('-')[1]
    mrxs_dir = f'{pref}_20{suff}'
    return mrxs_dir


def path_image_data(image_name, path_to_wsi, add_dir_mrxs=True):
    s = 0
    for ext in exts:
        if ext in image_name:
            id_name = image_name.split(ext)[0]
            break

    if not os.path.exists(os.path.join(path_to_wsi, image_name)) and (ext == '.mrxs' and add_dir_mrxs):
        mrxs_dir = get_mrxs_dir(image_name)
        path_to_image = os.path.join(path_to_wsi, mrxs_dir, image_name)
    else:
        path_to_image = os.path.join(path_to_wsi, image_name)

    if ext == '.scn':
        s = image_name[-1]
        id_name += f"__{s}"
        path_to_image = path_to_image.split('.scn')[0] + '.scn'

    return s, ext, id_name, path_to_image


def is_supported(wsi_path):
    for ext in exts:
        if wsi_path.endswith(ext):
            return True
    return False


exts = ['.ndpi', '.svs', '.scn', '.mrxs',
        '.ome.tif', '.ome.tiff',
        '.tif', '.tiff',
        ]


def get_reader_type(wsi_path):
    if is_supported(wsi_path):
        if wsi_path.endswith('.ndpi'):
            return ReaderType.NDPI
        elif wsi_path.endswith('.svs'):
            return ReaderType.SVS
        elif wsi_path.endswith('.scn'):
            return ReaderType.SCN
        elif wsi_path.endswith('.mrxs'):
            return ReaderType.MRXS
        elif wsi_path.endswith(('.ome.tif', '.ome.tiff')):
            return ReaderType.OME_TIFF
        elif wsi_path.endswith(('.tif', '.tiff')):
            return ReaderType.TIFF
    else:
        return ReaderType.NONE
