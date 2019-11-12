Implementation
===============

Start with the configuration:

.. toctree::
   :maxdepth: 3
   
   source/config
  
You can run the code by typing::

	$ python runme.py
	
``runme.py`` calls the main functions of the code, which are explained in the following sections.

.. toctree::
   :maxdepth: 3
   
   source/initialization 

Helping functions for the models are included in ``generate_intermediate_files.py``, ``correction_functions.py``, ``spatial_functions.py``, and ``input_maps.py``.

.. toctree::
   :maxdepth: 3
   
   source/generate_intermediate_files
   source/correction_functions
   source/spatial_functions
   source/input_maps
   
Utility functions as well as imported libraries are included in ``util.py``.

.. toctree::
   :maxdepth: 3
   
   source/util

Finally, the module ``generate_models.py`` contains formating functions that create the input files for the urbs and evrys models.

.. toctree::
   :maxdepth: 3
   
   source/generate_models