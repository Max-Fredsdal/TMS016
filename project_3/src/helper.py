### Helper functions for the project
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.special import kv, gamma
from sklearn.gaussian_process.kernels import Matern as SklearnMatern
import pandas as pd


def emp_variogram(D, data, N):
    """
    Computes a binned estimate of the semivariogram for spatial data.
    
    Parameters:
    D    : Measurement locations (n x 2) matrix or distance matrix (n x n).
    data : The measurements (n x 1 vector).
    N    : The number of bins.
    
    Returns:
    out  : A dictionary containing:
        - 'variogram' : The estimated semivariogram.
        - 'h'         : The center points for each bin.
        - 'N'         : Number of lags in each bin.
    """
    if D.shape[0] != D.shape[1]:
        D = squareform(pdist(D))  # Convert pairwise distances to a squareform matrix
    max_dist = np.max(D)  # Maximum distance
    d = np.linspace(0, max_dist, N)  # Bin edges (N bins)
    
    # Initialize output dictionary
    out = {'h': (d[1:] + d[:-1]) / 2,  # Bin centers
           'variogram': np.zeros(N-1),  # Variogram values
           'N': np.zeros(N-1)}  # Number of pairs in each bin
    
    # Compute the semivariogram for each bin
    for i in range(N-1):
        mask = (D > d[i]) & (D <= d[i+1])  # Find indices where distances are in the current bin
        I, J = np.where(mask)  # Find indices of pairs
        out['N'][i] = len(I)  # Number of pairs in the current bin
        
        if len(I) > 0:  # Avoid division by zero
            out['variogram'][i] = 0.5 * np.mean((data[I] - data[J]) ** 2)
    
    return out

import numpy as np
from scipy.special import kv, gamma

def matern_variogram(h, sigma, kappa, nu=1.5):
    """
    Computes the Matérn variogram.

    Parameters
    ----------
    h : array_like
        Lag distances.
    sigma : float
        Standard deviation (controls variance).
    kappa : float
        Inverse range (scale) parameter.
    nu : float, optional
        Smoothness parameter (default is 1.5).

    Returns
    -------
    output : ndarray
        Variogram values at lag distances h.
    """
    h = np.asarray(h)
    h = np.maximum(h, 1e-10)  # Prevent division by 0
    scaled_h = kappa * h
    factor = (2 ** (1 - nu)) / gamma(nu)
    matern = factor * (scaled_h ** nu) * kv(nu, scaled_h)
    cov = sigma ** 2 * matern
    cov[h == 0] = sigma ** 2  # Define covariance at zero distance

    output = sigma ** 2 - cov
    return output

# Loss function to fit the matern variogram to the empirical variogram
def fit_matern_variogram(emp_variogram_result, nu=1.5, initial_guess=[1.0, 1.0], bounds=[(1e-3, None), (1e-3, None)]):
    def loss(params):
        sigma, kappa = params
        pred = matern_variogram(emp_variogram_result['h'], sigma, kappa, nu)
        return np.sum((emp_variogram_result['variogram'] - pred) ** 2)
    
    res = minimize(loss, x0=initial_guess, bounds=bounds)
    return res.x  # returns (sigma_fit, kappa_fit)

def matern_covariance(coords1, coords2, sigma, kappa, nu=1):
    
    """
    Computes the Matern covariance matrix between two sets of coordinates.
    
    Parameters:
    coords1 : Array of coordinates (n x 2).
    coords2 : Array of coordinates (m x 2).
    sigma   : Variance parameter.
    kappa   : Scale parameter.
    nu      : Smoothness parameter.
    
    Returns:
    cov     : Covariance matrix.
    """

    matern_kernel = SklearnMatern(length_scale=1.0/kappa, nu=nu)
    cov = matern_kernel(coords1, coords2)
    return sigma**2 * cov


def gls_estimate(X, y, K):
    """
    Performs Generalized Least Squares (GLS) estimation.
    
    Parameters:
    X : ndarray of shape (n_samples, n_features)
        Design matrix (covariates).
    y : ndarray of shape (n_samples,)
        Observed responses.
    K : ndarray of shape (n_samples, n_samples)
        Covariance matrix of the observations.
    
    Returns:
    beta_hat : ndarray of shape (n_features,)
        Estimated regression coefficients.
    residuals : ndarray of shape (n_samples,)
        Residuals of the model.
    L : ndarray of shape (n_samples, n_samples)
        Cholesky decomposition of the covariance matrix.
    """
    # Cholesky decomposition for numerical stability
    L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))  # Regularize for invertibility

    # Solve K⁻¹y and K⁻¹X efficiently using Cholesky
    Kinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Kinv_X = np.linalg.solve(L.T, np.linalg.solve(L, X))

    # Compute (Xᵀ K⁻¹ X)⁻¹ Xᵀ K⁻¹ y
    Xt_Kinv_X = X.T @ Kinv_X
    Xt_Kinv_y = X.T @ Kinv_y
    beta_hat = np.linalg.solve(Xt_Kinv_X, Xt_Kinv_y)

    residuals = y - X @ beta_hat

    return beta_hat, residuals, L


def predict_missing_data(L, residuals, K_mo, X_miss, beta_hat):
    """
    Predicts the missing data using the kriging and trend components.
    
    Parameters:
    ----------
    L : ndarray
        Cholesky decomposition of the covariance matrix.
    residuals : ndarray
        Residuals from kriging system.
    K_mo : ndarray
        Covariance between missing and observed data points.
    X_miss : ndarray
        Design matrix for missing data.
    beta_hat : ndarray
        Estimated parameters (beta).
        
    Returns:
    -------
    r_m_hat : ndarray
        Predictions for missing data.
    """
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, residuals))
    kriging_part = K_mo @ alpha
    trend_part = X_miss @ beta_hat
    r_m_hat = trend_part + kriging_part

    return r_m_hat


def read_data_from_csv(path_obs, path_miss, covariates=None):
    import numpy as np
    import pandas as pd

    # Load data
    data_obs = pd.read_csv(path_obs)
    data_miss = pd.read_csv(path_miss)


    # Extract training data (observed)
    points_obs = data_obs[['latitude', 'longitude']].values
    values_obs = data_obs['price'].values

    # Handle covariates: if provided, extract them; if not, return empty 2D array
    if covariates is not None and len(covariates) > 0:
        covariates_obs = data_obs[covariates].values
    else:
        covariates_obs = np.empty((len(data_obs), 0))  # shape (n_samples, 0)

    # Extract prediction points (missing)
    points_miss = data_miss[['latitude', 'longitude']].values

    # Handle covariates: if provided, extract them; if not, return empty 2D array
    if covariates is not None and len(covariates) > 0:
        covariates_miss = data_miss[covariates].values
    else:
        covariates_miss = np.empty((len(data_miss), 0))  # shape (n_samples, 0)

    return points_obs, values_obs, covariates_obs, points_miss, covariates_miss