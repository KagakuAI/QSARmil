
QSARmil - Molecular Multi-Instance Machine Learning
============================================================
``QSARmil`` is a package for designing pipelines for building QSAR models with multi-instance machine learning algorithms.

Introduction
------------
Molecules are complex, dynamic objects that can exist in different molecular forms
(conformations, tautomers, stereoisomers, protonation states, etc.) or consist of several molecular fragments,
and often it is not known which molecular form/fragment is responsible for the observed physicochemical and
biological properties of a given molecule. Multi-instance machine learning (MIL) is an efficient approach
for solving problems where objects under study cannot be uniquely represented by a single instance,
but rather by a set of multiple alternative instances.


.. image:: docs/fgr_1.png
   :width: 800px

**Polymorphism ambiguity.** This type of ambiguity arises when a molecule can be represented by alternative instances,
such as conformations, tautomers, protonation states, etc.

**Part-to-whole ambiguity.** This type of ambiguity arises when a molecule can be represented by
as a set of atoms or fragments.

**Segment-to-sequence ambiguity.** Biological molecules (RNA, DNA, and proteins) can be represented by
multiple alternative segments/subsequences, and often only a particular segment of a sequence is responsible
for the biological function of a whole molecule.
You can find the pipelines for modelling these tasks in a separate project, `SEQmil <https://github.com/KagakuAI/SEQmil>`_ .


Installation
------------

``QSARmil`` can be installed using conda/mamba package managers.

.. code-block:: bash

    git clone https://github.com/KagakuAI/QSARmil.git
    conda env create -f QSARmil/conda/qsarmil_linux.yml
    conda activate qsarmil

The installed ``QSARmil`` environment can then be added to the Jupyter platform:

.. code-block:: bash

    conda install ipykernel
    python -m ipykernel install --user --name qsarmil --display-name "qsarmil"


Quick start
------------

See the examples of ``QSARmil`` application for different tasks in the `tutorial collection <tutorials>`_ .