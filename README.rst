
QSARmil - molecular multi-instance machine learning
============================================
``QSARmil`` is a package for designing pipelines for building QSAR models with multi-instance machine learning algorithms.

Installation
------------

``QSARmil`` can be installed using conda/mamba package managers.

To install `QSARmil``, first clone the repository and move the package directory:

.. code-block:: bash

    git clone https://github.com/KagakuAI/QSARmil.git
    conda env create -f QSARmil/conda/qsarmil.yaml
    conda activate qsarmil

The installed ``QSARmil`` environment can then be added to the Jupyter platform:

.. code-block:: bash

    conda install ipykernel
    python -m ipykernel install --user --name qsarmil --display-name "qsarmil"


Quick start
------------

See the `tutorial <tutorials/Tutorial_1_Pipeline.ipynb>`_ .
