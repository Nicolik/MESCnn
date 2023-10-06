# MESCnn Installation
We tested the following instructions to install all the dependencies 
required to run the MESCnn pipeline on *Ubuntu 20.04*. 
Please note that it should work fine also on other operating systems.
Though, some Python dependencies, like [Detectron2](https://github.com/facebookresearch/detectron2), 
may be more difficult to set-up on a Windows system.
 

## System Dependencies
The following system dependencies are required:

1. Install Java Runtime Environment (JRE) and Java Development Kit (JDK):
```
sudo apt install default-jdk
sudo apt install default-jre
```

2. Install [Openslide](https://openslide.org):
```
sudo apt-get install openslide-tools
```

3. Download and install [QuPath](https://github.com/qupath/qupath/releases).


## Python Dependencies
You may want to have the Python dependencies installed inside 
a virtual environment.
We recommend the adoption of [Anaconda](https://www.anaconda.com/),
since we used it in all our experiments.

As first thing, you should create and activate an Anaconda virtual environment,
please note that we used *Python 3.7* throughout our tests:
```
conda create --name MESCnn-Env python=3.7
conda activate MESCnn-Env
```

Then, you need to install the required Python packages.
Due to a problem in `functools`, we suggest to install first the
two following packages:
```
pip install numpy cached-property
```

Then, install PyTorch with CUDA support:
```
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
```

Lastly, install all the other requirements:
```
pip install -r requirements.txt
```

At this point, install [Detectron2](https://github.com/facebookresearch/detectron2):
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

To solve the aforementioned problem in `functools`, 
edit the file `detectron2/utils/events.py` of the installed *Detectron2* library, 
by replacing `from functools import cached_property` with 
`from cached_property import cached_property`.


## Notes
- If *Paquo* cannot find the *QuPath* directory, you may need to add a `.paquo.toml` 
file, as reported by [Paquo documentation](https://paquo.readthedocs.io/en/latest/configuration.html).
In this case, create such file in the root directory of the MESCnn cloned repository,
and put the following lines, setting the `qupath_dir` as appropriate:
```
qupath_dir="path/to/your/qupath/installation"
MOCK_BACKEND=false
```


- If some local modules are not found and the pipeline raise the
`ModuleNotFoundError` for those modules, add the root directory of the project to
 the Python path:
```
export PYTHONPATH=.
```

- If you get the following error:
`OSError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.`,
try to run the following line in terminal:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```
