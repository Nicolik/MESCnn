import os
import zipfile
from huggingface_hub import hf_hub_download


def download_detector(model_name, config_dir):
    return hf_hub_download(
        repo_id="MESCnn/MESCnn",
        filename=f"mescnn/detection/logs/{model_name}/{config_dir}/output/model_final.pth",
        token="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",
        local_dir='.',
        local_dir_use_symlinks=False,
        force_download=True,
    )


def download_slide(slide_name, local_dir):
    repo_id = "MESCnn/MESCnn-Sample-Data"
    filename = f"WSI/{slide_name}"
    return download_dataset(repo_id, filename, local_dir, do_unzip=False)


def download_project(local_dir):
    repo_id = "MESCnn/MESCnn-Sample-Data"
    filename = f"QuPathProject-NoAnnotations.zip"
    return download_dataset(repo_id, filename, local_dir, do_unzip=True)


def download_dataset(repo_id, filename, local_dir, do_unzip=False):
    print(f"Attempting to download {filename} from {repo_id}")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        token="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",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        force_download=True,
    )
    if do_unzip:
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        os.remove(zip_path)
        print(f"Unzip done! Check: {local_dir}")
    return zip_path


def sanitize_qupath_project(qp_dir):
    from paquo.projects import QuPathProject
    with QuPathProject(qp_dir, mode='a+') as qp:
        print(f"Created Project {qp.name}!")

        for i, image in enumerate(qp.images):
            print(f"{i} URI: {image.uri}")

        qp.update_image_paths(try_relative=True)
        print("Updated Image Paths!")

        for i, image in enumerate(qp.images):
            print(f"{i} URI: {image.uri}")
