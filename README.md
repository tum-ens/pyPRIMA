# generate-models
A pre-processing tool to automate the creation of energy system models using a common database

[![Documentation Status](https://readthedocs.org/projects/generate-models/badge/?version=latest)](http://generate-models.readthedocs.io/en/latest/?badge=latest)

## Features
* Aggregation of input data for any user-defined regions provided as a shapefile
* Automation of the pre-processing to document assumptions and avoid human errors
* Cleaning of raw input data and creation of model-independent intermediate files
* Adaptation to the intermediate files to the models urbs and evrys as of version |version|

## Applications
This code is useful if:

* You want to create different models using the same input database, but different model regions
* You want to harmonize the assumptions used in different model frameworks (for comparison and/or linking)
* You want to generate many models in a short amount of time with fewer human errors
