# pyPRIMA
**py**thon **PR**eprocessing of **I**nputs for **M**odel fr**A**meworks: a tool to automate the creation of energy system models using a common database

[![Documentation Status](https://readthedocs.org/projects/pyPRIMA/badge/?version=latest)](http://pyprima.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors)

## Features
* Aggregation of input data for any user-defined regions provided as a shapefile
* Automation of the pre-processing to document assumptions and avoid human errors
* Cleaning of raw input data and creation of model-independent intermediate files
* Adaptation of the intermediate files to the models urbs and evrys

## Applications
This code is useful if:

* You want to create different models using the same input database, but different model regions
* You want to harmonize the assumptions used in different model frameworks (for comparison and/or linking)
* You want to generate many models in a short amount of time with fewer human errors
