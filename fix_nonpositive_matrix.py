# -*- coding: utf-8 -*-
"""
Used to repair non-positive definite correlation matrices.
The eigenvalues of corr matrices must be positive.
Negative eigenvalues result from numerical error (i.e., -1.22e-16); this script helps repair those.

Source: https://pyportfolioopt.readthedocs.io/en/latest/_modules/pypfopt/risk_models.html
"""

import warnings
import numpy as np
import pandas as pd


def _is_positive_semidefinite(matrix):
    """
    Helper function to check if a given matrix is positive semidefinite.
    Any method that requires inverting the covariance matrix will struggle
    with a non-positive semidefinite matrix

    :param matrix: (covariance) matrix to test
    :type matrix: np.ndarray, pd.DataFrame
    :return: whether matrix is positive semidefinite
    :rtype: bool
    """
    try:
        # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
    """
    Check if a covariance matrix is positive semidefinite, and if not, fix it
    with the chosen method.

    The ``spectral`` method sets negative eigenvalues to zero then rebuilds the matrix,
    while the ``diag`` method adds a small positive value to the diagonal.

    :param matrix: raw covariance matrix (may not be PSD)
    :type matrix: pd.DataFrame
    :param fix_method: {"spectral", "diag"}, defaults to "spectral"
    :type fix_method: str, optional
    :raises NotImplementedError: if a method is passed that isn't implemented
    :return: positive semidefinite covariance matrix
    :rtype: pd.DataFrame
    """
    if _is_positive_semidefinite(matrix):
        return matrix

    warnings.warn(
        "The covariance matrix is non positive semidefinite. Amending eigenvalues."
    )

    # Eigendecomposition
    q, V = np.linalg.eigh(matrix)

    if fix_method == "spectral":
        # Remove negative eigenvalues
        q = np.where(q > 0, q, 0)
        # Reconstruct matrix
        fixed_matrix = V @ np.diag(q) @ V.T
    elif fix_method == "diag":
        min_eig = np.min(q)
        fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
    else:
        raise NotImplementedError("Method {} not implemented".format(fix_method))

    if not _is_positive_semidefinite(fixed_matrix):  # pragma: no cover
        warnings.warn(
            "Could not fix matrix. Please try a different risk model.", UserWarning
        )

    # Rebuild labels if provided
    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix
