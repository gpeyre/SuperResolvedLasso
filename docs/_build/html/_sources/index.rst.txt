SR-Lasso: Super-resolved Lasso 
===================================

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:
   :caption: Getting started

   vignettes/*

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API

   srlasso


This is a Python package for the article "Super-resolved Lasso" by Clarice Poon and Gabriel Peyr\'e. The Super-resolved Lasso is a method for recovering sparse measures from low resolution measurements. The method works on a discrete grid but recovers both the amplitudes and the 'off-the-grid' shift. 

SRLasso package description
-------------------------------

- **continuous_BP.py**: implementation of SR Lasso and continuous basis pursuit.

- **Lasso.py**: VarPro implementation of Lasso solvers. It makes use of a Hadamard factorization of the solution and utilizes a LBFGS solver.

- **mmd.py**: Maximum mean discrepancy for evaluating distance between two sparse measures.

- **operators.py**:  implements Fourier, Laplace and Gaussian operators

- **helper.py**: useful plotting functions


Jupyter Notebooks
-----------------

- **1D-Fourier.ipynb** : Demonstrates SR-Lasso in the case where the measurement operator is the sample Fourier transform.

- **2D-Tensor.ipynb** :  Demonstrates SR-Lasso in the case where the measurement operator is a 2D operator that is separable
      
- **3D-Tensor.ipynb** : Demonstrates SR-Lasso in the case where the measurement operator is a 3D operator that is separable

- **nD-comparison.ipynb** : Compares SR-Lasso with Lasso for dimension n problems. Reproduces the figures in our paper.

- **certificate.ipynb**: Plots the certificates for SR-Lasso and continuous basis pursuit.


Citation
--------

.. code-block:: bibtex

   @article{poon2023super,
   title={Super-resolved Lasso},
   author={Poon, Clarice and Peyr{\'e}, Gabriel},
   journal={arXiv preprint arXiv:2311.09928},
   year={2023}
   }



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
