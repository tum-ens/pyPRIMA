User manual
===========

Installation
------------

.. NOTE:: We assume that you are familiar with `git <https://git-scm.com/downloads>`_ and `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_.

First, clone the git repository in a directory of your choice using a Command Prompt window::

	$ ~\directory-of-my-choice> git clone https://github.com/tum-ens/pyPRIMA.git

We recommend using conda and installing the environment from the file ``pyPRIMA.yml`` that you can find in the repository. In the Command Prompt window, type::

	$ cd pyPRIMA\env\
	$ conda env create -f pyPRIMA.yml

Then activate the environment::

	$ conda activate pyPRIMA

In the folder ``code``, you will find multiple files:

.. tabularcolumns:: |l|l|

+---------------------------------------+------------------------------------------------------------------------------------+
| File                                  | Description                                                                        |
+=======================================+====================================================================================+
| config.py                             | used for configuration, see below.                                                 |
+---------------------------------------+------------------------------------------------------------------------------------+
| runme.py                              | main file, which will be run later using ``python runme.py``.                      |
+---------------------------------------+------------------------------------------------------------------------------------+
| lib\\initialization.py                | used for initialization.                                                           |
+---------------------------------------+------------------------------------------------------------------------------------+
| lib\\input_maps.py                    | used to generate input maps for the scope.                                         |
+---------------------------------------+------------------------------------------------------------------------------------+
| lib\\generate-models.py               | used to generate the model files from intermediate files.                          |
+---------------------------------------+------------------------------------------------------------------------------------+
| lib\\generate_intermediate_files.py   | used to generate intermediate files from raw data.                                 |
+---------------------------------------+------------------------------------------------------------------------------------+
| lib\\spatial_functions.py             | contains helping functions related to maps, coordinates and indices.               |
+---------------------------------------+------------------------------------------------------------------------------------+
| lib\\correction_functions.py          | contains helping functions for data correction/cleaning.                           |
+---------------------------------------+------------------------------------------------------------------------------------+
| lib\\util.py                          | contains minor helping functions and the necessary python libraries to be imported.|
+---------------------------------------+------------------------------------------------------------------------------------+

config.py                                                                                           
---------
This file contains the user preferences, the links to the input files, and the paths where the outputs should be saved.
The paths are initialized in a way that follows a particular folder hierarchy. However, you can change the hierarchy as you wish.

.. toctree::
   :maxdepth: 3
   
   source/config
   
runme.py
--------
``runme.py`` calls the main functions of the code:

.. literalinclude:: ../code/runme.py
   :language: python
   :linenos:
   :emphasize-lines: 10-19,22-28,31-32

Recommended input sources
-------------------------
Load time series for countries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`ENTSO-E <https://www.entsoe.eu/data/power-stats/>`_ publishes (or used to publish - the service has been discontinued as of November 2019) hourly load profiles
for each country in Europe that is part of ENTSO-E.

Sectoral load profiles
^^^^^^^^^^^^^^^^^^^^^^
The choice of the load profiles is not too critical, since the sectoral load profiles will be scaled according to their shares in the
yearly demand, and their shapes edited to match the hourly load profile. Nevertheless, examples of load profiles for Germany
can be obtained from the `BDEW <https://www.bdew.de/energie/standardlastprofile-strom/>`_.

Sector shares (demand)
^^^^^^^^^^^^^^^^^^^^^^
The sectoral shares of the annual electricity demand can be obtained from `Eurostat <https://ec.europa.eu/eurostat>`_.
The table reference is *nrg_105a*. Follow these instructions to obtain the file as needed by the code:

  * GEO: Choose all countries, but not EU
  * INDIC_NRG: Choose all indices
  * PRODUCT: Electrical energy (code 6000)
  * TIME: Choose years
  * UNIT: GWh
  * Download in one single csv file

Power plants and storage units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The powerplantmatching package within `FRESNA <https://github.com/FRESNA/powerplantmatching>`_ extracts a standardized power plant database that 
combines several other databases covering Europe.
In this repository, all non-renewable power plants, all storage units, and some renewable power plants (e.g. geothermal) are obtained from this database.
Since the capacities for most renewable technologies are inaccurate, they are obtained from another source (see below).

Renewable installed capacities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Renewable electricity capacity and generation statistics are obtained from the Query Tool of `IRENA <https://www.irena.org/Statistics/View-Data-by-Topic/Capacity-and-Generation/Query-Tool>`_.
The user has to create a query that includes all countries (but no groups of countries, such as continents), all technologies (but no groups of technology)
for a particular year and name the file ``IRENA_RE_electricity_statistics_allcountries_alltech_YEAR.csv``.
This dataset has a global coverage, however it does not provide the exact location of each project. The code includes an algorithm to distribute the
renewable capacities spatially.

Renewable potential maps
^^^^^^^^^^^^^^^^^^^^^^^^^
These maps are needed to distribute the renewable capacities spatially, since IRENA does not provide their exact locations.
You can use any potential maps, provided that they have the same extent as the geographic scope. Adjust the resolution parameters
in :mod:`config.py` accordingly. Such maps can be generated using the GitHub repository `tum-ens/pyGRETA <https://github.com/tum-ens/pyGRETA>`_.

Renewable time series
^^^^^^^^^^^^^^^^^^^^^^
Similarly, the renewable time series can be generated using the GitHub repository `tum-ens/pyGRETA <https://github.com/tum-ens/pyGRETA>`_.
This repository is particularly is the model regions are unconventional.

Transmission lines
^^^^^^^^^^^^^^^^^^
High-voltage power grid data for Europe and North America can be obtained from `GridKit <https://zenodo.org/record/47317>`_, which used
OpenStreetMap as a primary data source.
In this repository, we only use the file with the lines (links.csv).
In general, the minimum requirements for any data source are that the coordinates for the line vertices and the voltage are provided.

Other assumptions
^^^^^^^^^^^^^^^^^^
Currently, other assumptions are provided in tables filled by the modelers. Ideally, machine-readable datasets providing the
missing information are collected and new modules are written to read them and extract that information.

Recommended workflow
--------------------

The script is designed to be modular and split into three main modules: :mod:`lib.correction_functions`, :mod:`lib.generate_intermediate_files`, and :mod:`lib.generate_models`. 

.. WARNING:: The outputs of each module serve as inputs to the following module. Therefore, the user will have to run the script sequentially.

The recommended use cases of each module will be presented in the order in which the user will have to run them. 

1. :ref:`correctionFunctions`
2. :ref:`generateIntermediateFiles`
3. :ref:`generateModels`

The use cases associated with each module are presented below.

It is recommended to thoroughly read through the configuration file `config.py` and modify the input paths and 
computation parameters before starting the `runme.py` script.
Once the configuration file is set, open the `runme.py` file to define what use case you will be using the script for.

.. _correctionFunctions:

Correction and cleaning of raw input data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each function in this module is designed for a specific data set (usually mentioned at the end of the function name).
The pre-processing steps include filtering, filling in missing values, correcting/overwriting erronous values, aggregating and disaggregating entries,
and deleting/converting/renaming the attributes.

At this stage, the obtained files are valid for the whole geographic scope, and do not depend on the model regions.

.. _generateIntermediateFiles:

Generation of intermediate files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The functions in this module read the cleaned input data, and adapts it to the model regions. They also expand the attributes
based on assumptions to cover all the data needs of all the supported models. The results are saved in individual CSV files that
are model-independent. These files can be shared with modelers whose models are not supported, and they might be able to adjust them
according to their model input requirements, and use them.

.. _generateModels:

Generation of model input files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here, the input files are adapted to the requirements of the supported model frameworks (currently urbs and evrys).
Input files as needed by the scripts of urbs and evrys are generated at the end of this step.