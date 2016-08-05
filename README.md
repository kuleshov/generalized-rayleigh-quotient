Generalized Rayleigh Quotient Iteration
=======================================

MATLAB code for the paper:

	V. Kuleshov, Fast algorithms for sparse principal componenent analysis 
	based on Rayleigh quotient iteration. Proceedings of the 30th International
	Conference on Machine Learning, Atlanta, GA, 2013.

Send feedback to [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/).

Contents
--------

`GRQI.m`: An implementation of Algorithm 2 in the ICML paper. Function `GRQI`
computes K sparse principal components using generalized Rayleigh quotient iteration.

`GPower.m`: An implementation of Algorithms 3 and 4 in the ICML paper. 
Function `GPower` computes K sparse principal components using the generalized
power method as implemented in the paper by Journee et al.

`SSVD.m`: An implementation of Algorithm 5 in the ICML paper. Function `SSVD` 
computes a pair of sparse singular vectors.

`CompareGRQI.m`: Script that comapres `GRQI` with `GPower` and generates a series of plots.
