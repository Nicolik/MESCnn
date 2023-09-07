import os
import json
import logging
import numpy as np
from paquo.projects import QuPathProject
import shapely
import pandas as pd
from detection.qupath.config import GlomerulusDetection, init_data_dict, MIN_AREA_BBOX_GLOMERULUS
from detection.qupath.paths import path_image_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert QuPathProject Annotations to JSON format')
    parser.add_argument('-e', '--export', type=str, help='path/to/export', required=True)
    parser.add_argument('-w', '--wsi-dir', type=str, help='path/to/wsi/dir', required=True)
    parser.add_argument('-q', '--qupath', type=str, help='path/to/qupath', required=True)
    args = parser.parse_args()
    export_dir = os.path.join(args.export, 'Temp', 'qu2json-output')
    path_to_wsi = args.wsi_dir
    path_to_qupath = args.qupath

    os.makedirs(export_dir, exist_ok=True)

    json_dataset = os.path.join(export_dir, 'dataset_detectron2.json')
    csv_rois = os.path.join(export_dir, 'rois.csv')
    csv_wsi = os.path.join(export_dir, 'wsi.csv')

    names = set()
    dict_rois = init_data_dict()

    images_list_dict = []
    wsi_list = []

    with QuPathProject(path_to_qupath, mode='r') as qp:
        num_images = len(qp.images)
        print(f"opened project '{qp.name}' with {num_images} images")
        for i, image in enumerate(qp.images):
            if 'pas' not in image.image_name and 'PAS' not in image.image_name:
                print(f"check staining of image {image.image_name}")

            wsi_list.append(image.image_name)

            print(f"Processing image {image.image_name}, trying to load annotations...")
            if image.hierarchy.annotations:
                annotations = image.hierarchy.annotations
                shapes = {
                    'Cortex': [],
                    'Medulla': [],
                    'CapsuleOther': [],
                    'Glomerulus': []
                }

                print(f"Image {image.image_name} has {len(annotations)} annotations.")
                for a, annotation in enumerate(annotations):

                    # annotations are paquo.pathobjects.QuPathPathAnnotationObject instances
                    # their ROIs are accessible as shapely geometries via the .roi property
                    name = annotation.path_class.name if annotation.path_class else "none"
                    names.add(name)
                    print(f"> [I = {(i + 1):3d}/{len(qp.images):3d}] "
                          f"[A = {(a + 1):3d}/{len(annotations):3d}] class: {name}")
                    if type(annotation.roi) == shapely.geometry.polygon.Polygon:
                        if name in shapes:
                            shapes[name].append(annotation.roi)
                    else:
                        for roi in annotation.roi.geoms:
                            if name in shapes:
                                shapes[name].append(roi)

                s, ext, id_name, path_to_image = path_image_data(image.image_name, path_to_wsi, add_dir_mrxs=True)
                if os.path.exists(path_to_image):
                    tile_offset_x = 100
                    tile_offset_y = 100
                    for glomerulus in shapes['Glomerulus']:
                        xg, yg = glomerulus.exterior.coords.xy
                        xg, yg = np.array(xg, dtype=np.int32), np.array(yg, dtype=np.int32)

                        xt, yt = xg.min()-tile_offset_x, yg.min()-tile_offset_y
                        wt, ht = xg.max()-xg.min()+2*tile_offset_x, yg.max()-yg.min()+2*tile_offset_y
                        tile_id = f"glomerulus {id_name} [{xt}, {yt}, {wt}, {ht}]"
                        tile_filename = f"{tile_id}.jpeg"
                        print(f"Tile: [{xt}, {yt}, {wt}, {ht}]")

                        xg, yg = xg - xt, yg - yt

                        bbox_g = [int(xg.min()), int(yg.min()), int(xg.max()), int(yg.max())]
                        polygon_glomerulus = []
                        for xgi, ygi in zip(xg, yg):
                            polygon_glomerulus.extend([int(xgi), int(ygi)])

                        area_bbox = (yg.max()-yg.min()) * (xg.max()-xg.min())
                        if area_bbox > MIN_AREA_BBOX_GLOMERULUS:
                            dict_rois['image-id'].append(id_name)
                            dict_rois['filename'].append(tile_filename)
                            dict_rois['path-to-wsi'].append(path_to_image)
                            dict_rois['ext'].append(ext)
                            dict_rois['s'].append(s)
                            dict_rois['x'].append(xt)
                            dict_rois['y'].append(yt)
                            dict_rois['w'].append(wt)
                            dict_rois['h'].append(ht)

                            annotations_tile_glomerulus = {
                                'bbox': bbox_g,
                                'bbox_mode': 'BoxModeXYXY_ABS',
                                'category_id': int(GlomerulusDetection.GLOMERULUS),
                                'segmentation': [polygon_glomerulus]
                            }

                            image_dict = {
                                'file_name': tile_filename,
                                'height': int(ht),
                                'width': int(wt),
                                'image_id': tile_id,
                                'annotations': [annotations_tile_glomerulus]
                            }
                            images_list_dict.append(image_dict)
                        else:
                            logging.warning(f"image_id: {tile_id}, found area ({area_bbox})"
                                            f" lesser than {MIN_AREA_BBOX_GLOMERULUS}!")

    with open(json_dataset, 'w') as fp:
        json.dump(images_list_dict, fp)

    df_rois = pd.DataFrame(data=dict_rois)
    df_rois.to_csv(csv_rois)

    df_wsi = pd.DataFrame(data={'WSI': wsi_list})
    df_wsi.to_csv(csv_wsi)
