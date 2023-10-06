import os
from shapely.geometry import Polygon


def get_or_create_entry(qp, path_to_wsi):
    from paquo.images import QuPathImageType
    image_id = os.path.basename(path_to_wsi)
    entry = None
    print(f"[get_or_create_entry] image_id: {image_id}")
    print(f"[get_or_create_entry] images: {qp.images}")
    for qpi in qp.images:
        if image_id == qpi.image_name:
            entry = qpi
        elif image_id.endswith(".scn") and (image_id + " - Series 1") == qpi.image_name:
            entry = qpi
    print(f"[get_or_create_entry] after search, entry = {entry}")
    if not entry:
        entry = qp.add_image(path_to_wsi, image_type=QuPathImageType.BRIGHTFIELD_H_E, allow_duplicates=False)
        print(f"[get_or_create_entry] added entry: {entry}!")
    return entry


def pickle2qu(path_to_wsi, ddict, qupath_project_dir):
    from paquo.projects import QuPathProject
    from paquo.classes import QuPathPathClass

    with QuPathProject(qupath_project_dir, mode='a+') as qp:
        print(f"Created Project {qp.name}!")

        new_classes = {
            "Glomerulus": QuPathPathClass(name="Glomerulus", color="#0000ff"),
        }
        path_classes = [new_classes[key] for key in new_classes]
        qp.path_classes = path_classes

        entry = get_or_create_entry(qp, path_to_wsi)

        print(f"[json2qu] ddict.keys(): {ddict.keys()}")
        print(f"[json2qu] type(entry): {type(entry)}")

        annotations_polygonal = {}

        for i, poly_coord in enumerate(ddict["Glomerulus"]):
            annotations_polygonal[f"Glomerulus_{i}"] = Polygon(poly_coord)

        for name, roi in annotations_polygonal.items():
            entry.hierarchy.add_annotation(roi=roi, path_class=new_classes['Glomerulus'])

        print(f"[json2qu] done. Please look at {qp.name} in QuPath.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert pickle Annotations to QuPath Project')
    parser.add_argument('-w', '--wsi', type=str, help='path/to/wsi', required=True)
    parser.add_argument('-p', '--pickle', type=str, help='path/to/pickle', required=True)
    parser.add_argument('-q', '--qupath', type=str, help='path/to/qupath', required=True)

    args = parser.parse_args()

    import pickle
    with open(args.pickle, "rb") as fp:
        ddict = pickle.load(fp)
    print(f"Calling json2qu for wsi: {args.wsi} on qupath project: {args.qupath}")
    pickle2qu(args.wsi, ddict, args.qupath)
