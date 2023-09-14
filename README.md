# MESCnn
This repository contains an end-to-end pipeline, which we named *MESCnn* 
(*MESC* classification by *n*eural *n*etwork), for glomerular Oxford classification of 
whole slide images.

The Oxford Classification for IgA nephropathy is the most successful example of an
evidence-based nephropathology classification system. 

The aim of our study was to replicate the glomerular components of Oxford scoring with an end-to-end 
deep learning pipeline that involves automatic glomerular segmentation
followed by classification for mesangial hypercellularity (M), 
endocapillary hypercellularity (E), segmental sclerosis (S) and active crescents (C).

If you find this repository useful for your research, please cite our paper in *Computer Methods and Programs in Biomedicine*:
```
@article{ALTINI2023107814,
title = {Performance and Limitations of a Supervised Deep Learning Approach for the Histopathological Oxford Classification of Glomeruli with IgA Nephropathy},
journal = {Computer Methods and Programs in Biomedicine},
pages = {107814},
year = {2023},
issn = {0169-2607},
doi = {https://doi.org/10.1016/j.cmpb.2023.107814},
url = {https://www.sciencedirect.com/science/article/pii/S0169260723004807},
author = {Nicola Altini and Michele Rossini and Sándor Turkevi-Nagy and Francesco Pesce and Paola Pontrelli and Berardino Prencipe and Francesco Berloco and Surya Seshan and Jean-Baptiste Gibier and Anibal Pedraza Dorado and Gloria Bueno and Licia Peruzzi and Mattia Rossi and Albino Eccher and Feifei Li and Adamantios Koumpis and Oya Beyan and Jonathan Barratt and Huy Quoc Vo and Chandra Mohan and Hien Van Nguyen and Pietro Antonio Cicalese and Angela Ernst and Loreto Gesualdo and Vitoantonio Bevilacqua and Jan Ulrich Becker},
}
```

## Installation
In order to install the dependencies required to run MESCnn, check [INSTALL.md](INSTALL.md).

## Associated Data
The source code is associated with:
- Trained Models for both segmentation and classification stages. 
Available on [Hugging Face](https://huggingface.co/MESCnn/MESCnn).

- Sample WSIs to replicate the pipeline.
Available on [Hugging Face](https://huggingface.co/datasets/MESCnn/MESCnn-Sample-Data).

## Usage of end-to-end pipeline
Two examples of usage of the end-to-end pipeline are reported:
- Replicate the pipeline from WSIs input. The example is reported in `run_wsi_tif.py`.
Please note that the code will perform automatic download of sample WSIs and weights of trained models.

- Replicate the pipeline from an existing QuPath project. The example is reported in `run_project_tif.py`.
Please note that the code will perform automatic download of an existing QuPath project,
sample WSIs and weights of trained models.

### Operations
The following operations are involved in the pipeline:

1) Tile the WSI. See `detection/qupath/tile.py` for more information.
2) Segment the tiled WSI. See `detection/qupath/segment.py` for more information.
Internally, it will convert the pickle Detectron2 annotations
to a QuPath project. See `detection/qupath/pkl2qu.py` for more information.
3) Eventually, manually revise the generated QuPath project.
4) Export the QuPath project annotations to JSON.
See `detection/qupath/qu2json.py` for more information.
5) Generate glomerular crops from JSON annotations.
See `detection/qupath/json2exp.py` for more information.
6) Perform classification of glomerular crops and 
Oxford classification at WSI-level. 
See `classification/inference/mesc/classify.py` for more information.

If you are interested in tiling and segmenting an entire QuPath project,
see `detection/qupath/segment_project.py` for more details.
