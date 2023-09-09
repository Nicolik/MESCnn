# MESCnn Installation
To install all libraries and tools needed to run the MESCnn pipeline on Ubuntu, follow these instructions.


## Requirements
The following packages are required:
1. Install JDK and JRE with `sudo apt install default-jdk` and `sudo apt install default-jre`
2. Install Openslide from https://openslide.org/download/, using `sudo apt-get install openslide-tools`
3. Download QuPath from https://github.com/qupath/qupath/releases following the related instructions


## Python Installation
Create a virtual Enviroment and activate it. We reccommend the use of Conda:
1. conda create --name name_env python=3.7
2. conda activate name_env

Due to dependecies issues, first install the following libraries:
- pip install numpy cached-property

then run

- pip install -r requirements.txt

Install detectron2 using the following command:
- python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


In detecton2/utils/events change:
- `from functools import cached_property` with `from cached_property import cached_property`

Finally, edit `.paquo.toml` file by inserting QuPath path in qupath_dir variable: 
`qupath_dir = "path_to_QuPath"`

### Note
1. If the classification and dection package are not detected, before lauch the pipeline scripts, run in terminal:
- `export PYTHONPATH=.`

2. If you get the following error:
`OSError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.`,
run in terminal:
- `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7`
