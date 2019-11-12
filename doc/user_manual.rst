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
| lib\generate-models.py              | Missing                                                                   |
+-------------------------------------+---------------------------------------------------------------------------+
| lib\generate_intermediate_files.py  | Missing                                                                   |
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


Recommended workflow
--------------------