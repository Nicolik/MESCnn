import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

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
