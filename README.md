# Super-resolved Lasso

This is a Python package for the article `Super-resolved Lasso' by Clarice Poon and Gabriel Peyr\'e.
The Super-resolved Lasso is a method for recovering sparse measures from low resolution measurements. The method works on a discrete grid but recovers both the amplitudes and the 'off-the-grid' shift. 

## Main module:

1. **continuous_BP.py**: implementation of SR Lasso and continuous basis pursuit.


## Jupyter Notebooks:

1. **1D-Fourier.ipynb** : Demonstrates SR-Lasso in the case where the measurement operator is the sample Fourier transform.

2. **2D-Tensor.ipynb** :  Demonstrates SR-Lasso in the case where the measurement operator is a 2D operator that is separable
      
3. **3D-Tensor.ipynb** : Demonstrates SR-Lasso in the case where the measurement operator is a 3D operator that is separable

4. ** nD-comparison.ipynb** : Compares SR-Lasso with Lasso for dimension n problems. Reproduces the figures in our paper.

5. **certificate.ipynb**: Plots the certificates for SR-Lasso and continuous basis pursuit.


## Support Modules:

1. **Lasso.py**: VarPro implementation of Lasso solvers. It makes use of a Hadamard factorization of the solution and utilizes a LBFGS solver.

2. **mmd.py**: Maximum mean discrepancy for evaluating distance between two sparse measures.

3. **operators.py**:  implements Fourier, Laplace and Gaussian operators

4. **helper.py**: useful plotting functions