from helper import*

import numpy as np

######################## Controls ###############################################

plot_variogram = True  # Set to True to plot the empirical and fitted variograms
covs = ['latitude', 'longitude','distance_to_center_km','distance_to_station_km']
target_class = 'Hotel room' # ['Private room' 'Entire home/apt' 'Shared room' 'Hotel room']  # Target class for filtering

n_sample = 1500  # Number of pixels to sample (change this value)
nu = 1.5  # Smoothness parameter for the Matern variogram
nBins = 60 # Number of bins for the empirical variogram

filepath_obs = 'project_3/data/Airbnb_clean_dist_cent_sub.csv'  # Path to the CSV file
filepath_miss = 'project_3/data/manhattan_grid_points_with_dist.csv'  # Path to the CSV file with missing data

#################################################################################


def main():

    points_obs, values_obs, covariates_obs, points_miss, covariates_miss = read_data_from_csv(filepath_obs, filepath_miss, covs)

    n = len(points_obs)
    # values_obs = np.log(values_obs + 1) 

    sampled_idx = np.random.choice(n, size=n_sample, replace=False)
    
    points_sample = points_obs[sampled_idx]
    values_sample = values_obs[sampled_idx]
    covariates_sample = covariates_obs[sampled_idx]
    
    print(min(values_sample), max(values_sample))

    from sklearn.preprocessing import StandardScaler

    # Normalize covariates
    scaler = StandardScaler()
    covariates_sample = scaler.fit_transform(covariates_sample)
    covariates_miss = scaler.transform(covariates_miss)

    # Compute the empirical variogram from observed data
    emp_variogram_result = emp_variogram(points_sample, np.log(values_sample + 1) , N=nBins) # add this np.log(values_sample + 1)

    plt.plot(emp_variogram_result['h'], emp_variogram_result['variogram'], 'ko-')
    plt.title("Raw Empirical Variogram (log-space)")
    plt.xlabel("Lag Distance")
    plt.ylabel("Semivariance")
    plt.grid(True)
    plt.show()


    # Fit matern variogram to the empirical variogram using least squares
    sigma_fit, kappa_fit = fit_matern_variogram(emp_variogram_result, nu, initial_guess=[100.0, 0.1], bounds=[(1e-3, None), (1e-3, None)])
    print(f"kappa_fit: {kappa_fit}, sigma_fit: {sigma_fit}")

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
    K_oo = matern_covariance(points_sample, points_sample, sigma_fit, kappa_fit, nu)
    K_mo = matern_covariance(points_miss, points_sample, sigma_fit, kappa_fit, nu)

    # add nugget to K_oo for numerical stability
    K_oo += 1e-6 * np.eye(K_oo.shape[0])

    # Define feature matrix
    B_obs = np.hstack((np.ones((points_sample.shape[0], 1)), covariates_sample))
    B_miss = np.hstack((np.ones((points_miss.shape[0], 1)), covariates_miss))

    # Non general case to be removed
    # B_obs = np.hstack((np.ones((points_sample.shape[0], 1)), points_sample))
    # B_miss = np.hstack((np.ones((points_miss.shape[0], 1)), points_miss))


    beta_hat, residuals, L = gls_estimate(B_obs, values_sample, K_oo)

    r_m_hat = predict_missing_data(L, residuals, K_mo, B_miss, beta_hat)
    print(beta_hat)
    print(min(r_m_hat))

    # Plot predicted values at missing locations
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(points_miss[:, 1], points_miss[:, 0], c=r_m_hat, cmap='viridis', s=5)
    plt.colorbar(sc, label='Predicted Price')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Predicted Prices at Missing Locations')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    # reconstructed = np.zeros((n, m))
    # reconstructed[mask] = values_sample
    # reconstructed[~mask] = r_m_hat

    # # Plot the results
    # plt.figure(figsize=(15, 5))

    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title("Observed (Missing Data)")
    # plt.imshow(np.where(mask, image, np.nan), cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title("Reconstructed Image")
    # plt.imshow(reconstructed, cmap='gray')
    # plt.axis('off')

    # plt.show()

# def main():
#     points_obs, values_obs, covariates_obs, points_miss, covariates_miss = read_data_from_csv(
#         filepath_obs, filepath_miss, covs
#     )

#     n = len(points_obs)
    
#     # Log-transform the observed prices
#     values_obs_log = np.log1p(values_obs)

#     # Sample a subset
#     sampled_idx = np.random.choice(n, size=n_sample, replace=False)
#     points_sample = points_obs[sampled_idx]
#     values_sample_log = values_obs_log[sampled_idx]
#     covariates_sample = covariates_obs[sampled_idx]

#     # Normalize covariates
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     covariates_sample = scaler.fit_transform(covariates_sample)
#     covariates_miss = scaler.transform(covariates_miss)

#     # Compute empirical variogram in log space
#     emp_variogram_result = emp_variogram(points_sample, values_sample_log, N=nBins)
#     dff = pd.DataFrame(emp_variogram_result)
#     dff.to_csv("my_list.csv", index=False)


#     plt.plot(emp_variogram_result['h'], emp_variogram_result['variogram'], 'ko-')
#     plt.title("Raw Empirical Variogram (log-space)")
#     plt.xlabel("Lag Distance")
#     plt.ylabel("Semivariance")
#     plt.grid(True)
#     plt.show()

#     plt.plot(emp_variogram_result['h'], emp_variogram_result['N'], 'ro-')
#     plt.show()
#     # Fit Matérn variogram to the empirical one
#     sigma_fit, kappa_fit = fit_matern_variogram(emp_variogram_result, nu, initial_guess=[100.0, 0.1])
#     print(f"kappa_fit: {kappa_fit}, sigma_fit: {sigma_fit}")

#     if plot_variogram:
#         plt.figure()
#         plt.plot(emp_variogram_result['h'], emp_variogram_result['variogram'], 'ko', label="Empirical")
#         plt.plot(emp_variogram_result['h'], matern_variogram(emp_variogram_result['h'], sigma_fit, kappa_fit, nu), 'r-', label="Fitted Matérn")
#         plt.xlabel('Lag Distance')
#         plt.ylabel('Semivariance (log-space)')
#         plt.legend()
#         plt.title("Empirical vs Fitted Matern Variogram (log-space)")
#         plt.show()

#     # Covariance matrices
#     K_oo = matern_covariance(points_sample, points_sample, sigma_fit, kappa_fit, nu)
#     K_mo = matern_covariance(points_miss, points_sample, sigma_fit, kappa_fit, nu)
#     K_oo += 1e-6 * np.eye(K_oo.shape[0])  # Regularization

#     # Feature matrices
#     B_obs = np.hstack((np.ones((points_sample.shape[0], 1)), covariates_sample))
#     B_miss = np.hstack((np.ones((points_miss.shape[0], 1)), covariates_miss))

#     # GLS estimation and kriging (in log-space)
#     beta_hat, residuals, L = gls_estimate(B_obs, values_sample_log, K_oo)
#     r_m_hat_log = predict_missing_data(L, residuals, K_mo, B_miss, beta_hat)

#     # Back-transform to original price space
#     r_m_hat = np.expm1(r_m_hat_log)

#     print("Beta coefficients (log-space):", beta_hat)
#     print("Min predicted price:", np.min(r_m_hat))

#     # Plot predicted values
#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(points_miss[:, 1], points_miss[:, 0], c=r_m_hat, cmap='viridis', s=5)
#     plt.colorbar(sc, label='Predicted Price')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('Predicted Prices at Missing Locations')
#     plt.gca().set_aspect('equal')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    main()