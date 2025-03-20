
MI-PROP - a tool for predictive modelling
============================================
MI-PROP tool

Installation
------------

``MI-PROP`` can be installed using conda/mamba package managers.

To install ``MI-PROP``, first clone the repository and move the package directory:

.. code-block:: bash

    git clone https://github.com/dzankov/MI-PROP.git
    cd MI-PROP/

Next, create ``MI-PROP`` environment with ``miprop_env.yaml`` file:

.. code-block:: bash

    conda env create -f conda/miprop_env.yaml
    conda activate miprop

The installed ``MI-PROP`` environment can then be added to the Jupyter platform:

.. code-block:: bash

    conda install ipykernel
    python -m ipykernel install --user --name miprop --display-name "miprop"


Quick start
------------

Python interface



