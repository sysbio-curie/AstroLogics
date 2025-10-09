<img align="left" width="160" src="https://github.com/sysbio-curie/AstroLogics/raw/main/doc/images/Logo_full.svg" />

# An analysis framework for monotonous Boolean model ensemble

This is a repository of data, code and analyses of AstroLogics framework.
A step-by step tutorial can be found in the folder tutorial. Please have a look at our [tutorials](https://astrologics.readthedocs.io/en/latest/tutorials.html). 

## Overview
AstroLogics is a Python package designed for analysing monotonous Boolean model ensemble, a product of Boolean model synthesis from method such as [Bonesis](https://bnediction.github.io/bonesis/index.html).

Our framework includes two major processes 
1. Dynamical properties analysis : 
    - Calculated distance between models through probabilistic approxmition via [MaBoSS](https://github.com/sysbio-curie/MaBoSS).
2. Logical function evaluation : 
    - Features logical equation and identify key logical features between model clusters

<p align="center">
<img height="800" src="https://github.com/sysbio-curie/AstroLogics/raw/main/doc/images/Figure2_Overview_framework.png" />
<br>
<em> Overview of the framework showing the two major processes in the framework. <strong>Dynamics</strong>: dynamical properties analysis. <strong>Logics</strong>: Logical function evaluation </em>
</br>
</p>

## Getting Started
### Requirements (for AstroLogics)
- Python version 3.8 or greater
- Python's packages listed here:
    - pandas
    - numpy
    - scipy, sklearn
    - maboss
    - boolsim
    - bonesis
    - mpbn
### Installation 

There are several ways to install AstroLogics


#### PyPi

```
pip install astrologics
```

#### Conda
```
conda install -c colomoto astrologics
```

#### From source
First clone this directory:
```
git clone https://https://github.com/sysbio-curie/AstroLogics
```

Then install AstroLogics with pip
```
pip install AstroLogics
```


## Tutorials

Tutorials are available as Jupyter notebooks

### Run with Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sysbio-curie/AstroLogics/main?filepath=AstroLogics)


### Run locally with Docker
To run this notebook using the built docker image, run : 
```
docker run -p 8888:8888 -d sysbiocurie/astrologics
```

### Run locally with Conda
Creating the conda environment
```
conda env create --file environment.yml
```

To activate it : 
```
conda activate astrologics
```


To run the notebook: 
```
jupyter-lab
```

## Documentation

Our documentation is available on [ReadTheDocs](https://astrologics.readthedocs.io/)


## Citing AstroLogics
Coming soon
