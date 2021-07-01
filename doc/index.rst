========
pyPRIMA
========

.. figure:: img/pyPRIMA_logo.png
   :width: 20%
   :align: left
   :alt: pyPRIMA_logo

.. |br| raw:: html

   <br />

|br|

|br|
------------------------------------------------------------------------
python PReprocessing of Inputs for Model frAmeworks
------------------------------------------------------------------------

:Code developers: Kais Siala, Houssame Houmy
:Documentation authors:	Kais Siala		
:Maintainers: Kais Siala <kais.siala@tum.de>
:Organization: `Chair of Renewable and Sustainable Energy Systems <http://www.ens.ei.tum.de/en/homepage/>`_, Technical University of Munich
:Version: |version|
:Date: |today|
:License:
 The model code is licensed under the `GNU General Public License 3.0  <http://www.gnu.org/licenses/gpl-3.0>`_.  
 This documentation is licensed under a `Creative Commons Attribution 4.0 International <http://creativecommons.org/licenses/by/4.0/>`_ license. 

Features
--------
* Aggregation of input data for any user-defined regions provided as a shapefile
* Automation of the pre-processing to document assumptions and avoid human errors
* Cleaning of raw input data and creation of model-independent intermediate files
* Adaptation to the intermediate files to the models urbs and evrys as of version |version|

Applications
-------------
This code is useful if:

* You want to create different models using the same input database, but different model regions
* You want to harmonize the assumptions used in different model frameworks (for comparison and/or linking)
* You want to generate many models in a short amount of time with fewer human errors

Changes
--------
version 1.0.1
^^^^^^^^^^^^^^
* Adding logo for the tool.
* Minor fixes and improvements.

version 1.0.0
^^^^^^^^^^^^^^
This is the initial version.


Contents
--------
User manual
^^^^^^^^^^^^^

These documents give a general overview and help you get started from the installation to you first running model.

.. the following section contains the links to the other parts of the documentation, the 'maxdepth' component define how many sub sections should be displayed

.. toctree::
   :maxdepth: 3
   
   user_manual
  
Theoretical description 
^^^^^^^^^^^^^^^^^^^^^^^

Continue here if you want to understand the theoretical conception behind the generate load time series function.

.. toctree::
   :maxdepth: 2

   theory

Technical documentation
^^^^^^^^^^^^^^^^^^^^^^^
Continue here if you want to understand in detail the model implementation.

.. toctree::
   :maxdepth: 2
   
   implementation

Dependencies
------------

A list of the used libraries is available in the environment file:

.. literalinclude:: ../env/gen_mod.yml

Bibliography	
------------

.. toctree::
   :maxdepth: 1
   
   zref

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
