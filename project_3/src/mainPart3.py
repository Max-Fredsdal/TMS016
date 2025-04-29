from helper import*


######################## Controls ###############################################

plot_missing = False # Set to True to plot the missing data mask
plot_variogram = True  # Set to True to plot the empirical and fitted variograms

n_sample = 10000  # Number of pixels to sample (change this value)
nu = 1.5  # Smoothness parameter for the Matern variogram

filepath = 'project_3/data/housing_data.csv'  # Path to the CSV file

#################################################################################


def main():

    csv_file = filepath  # Replace with your CSV file path
    points, values, covariates = read_data_from_csv(csv_file)

    n = len(points)  # Number of points in the dataset

    sampled_idx = np.random.choice(n, size=n_sample, replace=False)
    
    observed_idx = points[sampled_idx]
    observed_values = values[sampled_idx]

    # Identify missing points (remaining points)
    missing_idx = np.delete(points, sampled_idx, axis=0)


    ### THIS CODE IS FOR PART 1 and is not generalizable to other data ############################
    x_titan = imageio.imread('project_3/data/titan.jpg')  # Update with the correct path
    x_titan_norm = x_titan / 255.0
    image = x_titan_norm
    n, m = image.shape

    n_sample = 10000  # Specify the exact number of pixels to sample (change this value)

    all_idx = np.array(np.unravel_index(np.arange(n * m), (n, m))).T  # Shape: (n*m, 2)

    sampled_idx = all_idx[np.random.choice(all_idx.shape[0], size=n_sample, replace=False)]

    mask = np.zeros((n, m), dtype=bool)
    mask[sampled_idx[:, 0], sampled_idx[:, 1]] = True

    observed_values = image[mask].flatten()

    observed_idx = np.argwhere(mask)
    missing_idx = np.argwhere(~mask)
    ################################################3


    # Visualize the mask
    if plot_missing:
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Masked Image ({n_sample} Pixels Sampled)")
        plt.imshow(np.where(mask, image, np.nan), cmap='gray')
        plt.axis('off')

        plt.show()

    # Compute the empirical variogram from observed data
    emp_variogram_result = emp_variogram(observed_idx, observed_values, N=100)

    # Fit matern variogram to the empirical variogram using least squares
    sigma_fit, kappa_fit = fit_matern_variogram(emp_variogram_result, nu, initial_guess=[1.0, 1.0], bounds=[(1e-3, None), (1e-3, None)])

    # Plot the fitted Matern variogram
    if plot_variogram:
        plt.figure()
        plt.plot(emp_variogram_result['h'], emp_variogram_result['variogram'], 'ko', label="Empirical")
        plt.plot(emp_variogram_result['h'], matern_variogram(emp_variogram_result['h'], sigma_fit, kappa_fit, nu), 'r-', label="Fitted Matérn")
        plt.xlabel('Lag Distance')
        plt.ylabel('Semivariance')
        plt.legend()
        plt.title("Empirical vs Fitted Matern Variogram")
        plt.show()

    # Calculate covariance matrix
    K_oo = matern_covariance(observed_idx, observed_idx, sigma_fit, kappa_fit, nu)
    K_mo = matern_covariance(missing_idx, observed_idx, sigma_fit, kappa_fit, nu)

    # add nugget to K_oo for numerical stability
    K_oo += 1e-6 * np.eye(K_oo.shape[0])

    # Define feature matrix
    B_obs = np.hstack((np.ones((observed_idx.shape[0], 1)), observed_idx, covariates[sampled_idx]))
    B_miss = np.hstack((np.ones((missing_idx.shape[0], 1)), missing_idx, covariates[~np.isin(points[:, 0], observed_idx[:, 0])]))

    # Non general case to be removed
    # B_obs = np.hstack((np.ones((observed_idx.shape[0], 1)), observed_idx))
    # B_miss = np.hstack((np.ones((missing_idx.shape[0], 1)), missing_idx))


    beta_hat, residuals, L = gls_estimate(B_obs, observed_values, K_oo)

    r_m_hat = predict_missing_data(L, residuals, K_mo, B_miss, beta_hat)
    
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

if __name__ == "__main__":
    main()