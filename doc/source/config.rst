Main configuration function
---------------------------

.. automodule:: config
   :members: configuration, general_settings
   
.. NOTE::
   Both *param* and *paths* will be updated in the code after running the function :mod:`config.configuration`.

.. NOTE::
   ``root`` points to the directory that contains all the inputs and outputs.
   All the paths will be defined relatively to the root, which is located in a relative position to the current folder.
   
The code differentiates between the geographic scope and the subregions of interest.
You can run the first part of the script ``runme.py`` once and save results for the whole scope, and then repeat the second part using different subregions within the scope.

.. automodule:: config
   :noindex:
   :members: scope_paths_and_parameters

.. NOTE::
   We recommend using a name tag that describes the scope of the bounding box of the regions of interest.
   For example, ``'Europe'`` and ``'Europe_without_Switzerland'`` will actually lead to the same output for the first part of the code.
   
User preferences
----------------

.. automodule:: config
   :noindex:
   :members: resolution_parameters
   
.. NOTE::
   As of version |version|, these settings should not be changed. Only MERRA-2 data can be used in the tool.
   Its spatial resolution is 0.5° of latitudes and 0.625° of longitudes. The high resolution is 15 arcsec in both directions.

.. automodule:: config
   :noindex:
   :members: load_parameters, renewable_time_series_parameters, grid_parameters, processes_parameters


Paths
------

.. automodule:: config
   :noindex:
   :members: assumption_paths, load_input_paths, renewable_time_series_paths, grid_input_paths, processes_input_paths, output_folders, output_paths, local_maps_paths
