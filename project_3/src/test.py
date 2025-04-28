import imageio
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import Matern

# 1. Load a synthetic image (you can replace this with your own image)
x_titan = imageio.imread('project_3/data/titan.jpg')  # Update with the correct path
x_titan_norm = x_titan / 255.0
image = x_titan_norm
n, m = image.shape

# 2. Simulate missing pixels (for example, randomly mask ~30% of the image)
mask = np.random.rand(n, m) > 0.7  # ~30% missing
observed_idx = np.argwhere(mask)
missing_idx = np.argwhere(~mask)
observed_values = image[mask]

# 3. Compute empirical variogram (based on distances between observed points)
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

# Compute the empirical variogram from observed data
emp_variogram_result = emp_variogram(observed_idx, observed_values, N=20)

# Plot the empirical variogram
plt.figure()
plt.plot(emp_variogram_result['h'], emp_variogram_result['variogram'], 'ko-', label="Empirical Variogram")
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Empirical Variogram')
plt.legend()
plt.show()

# 4. Fit Matern variogram to the empirical variogram using least squares
def matern_variogram(h, sigma, kappa, nu=1.5):
    """
    Computes the Matérn variogram based on the distance vector h.
    
    Parameters:
    h     : Distance vector.
    sigma : Variance.
    kappa : Scale parameter for the covariance function.
    nu    : Smoothness parameter.
    
    Returns:
    sv    : The semivariogram values.
    """
    matern_kernel = Matern(length_scale=1.0/kappa, nu=nu)
    cov = matern_kernel(h[:, None])  # Compute covariance matrix
    return sigma**2 * (1 - cov)

def loss(params):
    sigma, kappa = params
    pred = matern_variogram(emp_variogram_result['h'], sigma, kappa)
    return np.sum((emp_variogram_result['variogram'] - pred) ** 2)

# Minimize the loss function to fit the parameters
res = minimize(loss, x0=[1.0, 1.0], bounds=[(1e-3, None), (1e-3, None)])
sigma_fit, kappa_fit = res.x
print(f"Fitted sigma: {sigma_fit:.3f}, kappa: {kappa_fit:.3f}")

# Plot the fitted Matern variogram
plt.figure()
plt.plot(emp_variogram_result['h'], emp_variogram_result['variogram'], 'ko', label="Empirical Variogram")
plt.plot(emp_variogram_result['h'], matern_variogram(emp_variogram_result['h'], sigma_fit, kappa_fit), 'r-', label="Fitted Matern")
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.legend()
plt.title("Empirical vs Fitted Matern Variogram")
plt.show()

# 5. Build covariance matrices
def matern_covariance(coords1, coords2, sigma, kappa, nu=1.5):
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
    dists = cdist(coords1, coords2)
    matern_kernel = Matern(length_scale=1.0/kappa, nu=nu)
    cov = matern_kernel(dists)
    return sigma**2 * cov

print(f"Fitting covariance matrix...")
K_oo = matern_covariance(observed_idx, observed_idx, sigma_fit, kappa_fit)
K_mo = matern_covariance(missing_idx, observed_idx, sigma_fit, kappa_fit)

# 6. Solve kriging system
print(f"Solving kriging system...")
residuals_obs = observed_values  # assuming mean = 0 for now
r_m_hat = K_mo @ np.linalg.solve(K_oo, residuals_obs)

# 7. Reconstruct the image
reconstructed = np.zeros((n, m))
reconstructed[mask] = observed_values
reconstructed[~mask] = r_m_hat

# Plot the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Observed (Missing Data)")
plt.imshow(np.where(mask, image, np.nan), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(reconstructed, cmap='gray')
plt.axis('off')

plt.show()

