import os
import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

os.system("pip install numpy cached-property")
os.system("pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117")

setuptools.setup(
    name='MESCnn',
    version='0.1',
    description='MESC classification by neural network',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Nicolik/MESCnn',
    python_requires='>=3.7',
    packages=setuptools.find_packages(),
    install_requires=required,
)

if 'win' in sys.platform:
    os.system("pip install fvcore cloudpickle omegaconf pycocotools fairscale timm")
os.system("python -m pip install 'git+https://github.com/facebookresearch/detectron2.git")
