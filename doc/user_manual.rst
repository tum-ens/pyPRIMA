User manual
===========

Installation
------------

.. NOTE:: We assume that you are familiar with `git <https://git-scm.com/downloads>`_ and `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_.

First, clone the git repository in a directory of your choice using a Command Prompt window::

	$ ~\directory-of-my-choice> git clone https://github.com/tum-ens/generate-models.git

We recommend using conda and installing the environment from the file ``gen_mod.yml`` that you can find in the repository. In the Command Prompt window, type::

	$ cd generate-models\env\
	$ conda env create -f gen_mod.yml

Then activate the environment::

	$ conda activate gen_mod

In the folder ``code``, you will find multiple files:

.. tabularcolumns:: |p{3.7cm}|p{9cm}|

+-------------------------------------+---------------------------------------------------------------------------+
| File                                | Description                                                               |
+=====================================+===========================================================================+
| config.py                           | used for configuration, see below.                                        |
+-------------------------------------+---------------------------------------------------------------------------+
| runme.py                            | main file, which will be run later using ``python runme.py``.             |
+-------------------------------------+---------------------------------------------------------------------------+
| lib\initialization.py               | used for initialization.                                                  |
+-------------------------------------+---------------------------------------------------------------------------+
| lib\input_maps.py                   | used to generate input maps for the scope.                                |
+-------------------------------------+---------------------------------------------------------------------------+
| lib\generate-models.py              | used to generate the model files from intermediate files.                 |
+-------------------------------------+---------------------------------------------------------------------------+
| lib\generate_intermediate_files.py  | used to generate intermediate files from raw data.                        |
+-------------------------------------+---------------------------------------------------------------------------+
| lib\spatial_functions.py            | contains helping functions related to maps, coordinates and indices.      |
+-------------------------------------+---------------------------------------------------------------------------+
| lib\correction_functions.py         | contains helping functions for data correction/cleaning.                  |
+-------------------------------------+---------------------------------------------------------------------------+
| lib\util.py                         | contains minor helping functions and the necessary python libraries to be |
|                                     | imported.                                                                 |
+-------------------------------------+---------------------------------------------------------------------------+

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
   :emphasize-lines: 9-13,16-23,26-27

Recommended input sources
-------------------------
Load time series for countries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sectoral load profiles
^^^^^^^^^^^^^^^^^^^^^^

Sector shares (demand)
^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^



Transmission lines
^^^^^^^^^^^^^^^^^^
High-voltage power grid data for Europe and North America can be obtained from `GridKit <https://zenodo.org/record/47317>`_, which used
OpenStreetMap as a primary data source.
In this repository, we only use the file with the lines (links.csv).
In general, the minimum requirements for any data source are that the coordinates for the line vertices and the voltage are provided.


Other assumptions
^^^^^^^^^^^^^^^^^^


Recommended workflow
--------------------