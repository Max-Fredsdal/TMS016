% main.m
clear;

%% ##### Control parameters ################################
nSamples = 10000;  % LSE sample
pObs = 0.5; % Probability that a pixel is not observed
data = 0; % 1: Use Titan image, o.w. : use Rosetta image

bins = 100;
s_init = 1;
kappa_init = 1;
sigma_e_init = 1;
nu = 1;

kFactor = 1;
%% #########################################################

%% Initialize Images
x_rosetta = imread('data/rosetta.jpg');
x_titan = imread('data/titan.jpg');

% Apply grayscale and normalize color values to [0,1]
x_rosetta_gray_norm = double(rgb2gray(x_rosetta))./255;
x_titan_norm = double(x_titan)./255;


%% perturbe image
if data == 1
    image = x_titan_norm;
else
    image = x_rosetta_gray_norm;
end

[n, m] = size(image);
nPixels = n * m;
I_vec = image(:);

sample_idx = randperm(nPixels, nSamples);  % Sample nSamples unique indices
[row_sample, col_sample] = ind2sub([n, m], sample_idx);  % Get row and column coordinates

% Identify the indices for observed and missing pixels
idx = 1:nPixels;
obs_idx = idx(sample_idx);%find(observations)';  % Indices of observed pixels
miss_idx = idx;  % Indices of missing pixels
miss_idx(sample_idx) = [];

y_sample = I_vec(obs_idx);  % Get the observed pixel values for the sampled indices

% Used to plot the pertubed image
image_obs = NaN(n,m);
image_obs(obs_idx) = y_sample;

%% Estimate parameters
sample_coords = [row_sample(:), col_sample(:)];
[row_full, col_full] = ind2sub([n, m], (1:(n*m))');

B_0 = [ones(length(y_sample), 1), row_sample(:), col_sample(:)];
B = [ones(n*m, 1), row_full(:), col_full(:)];

beta_hat = B_0' * B_0 \ B_0' * y_sample;
mu_full = B * beta_hat;
mu_sample = mu_full(obs_idx);
mu_miss = mu_full(miss_idx);

residuals = y_sample - mu_sample;

% 
disp("Running emp_varioram")
emp = emp_variogram(sample_coords, residuals, bins);
lag = emp.h;
gamma_emp = emp.variogram;

theta_init = [s_init, kappa_init, nu, sigma_e_init];
disp("running cov_ls_est")
theta_est = cov_ls_est(residuals, 'matern', emp,struct('nu', 1));

%% PLOTS FOR 1.
figure;
plot(emp.h, emp.variogram, 'ko'); hold on
plot(emp.h, matern_variogram(emp.h, theta_est.sigma, theta_est.kappa, theta_est.nu, theta_est.sigma_e), 'r-')
legend('Empirical variogram', 'Fitted Matérn')
xlabel('Lag distance'); ylabel('Semivariance')
title('Variogram Fit')

disp("Kriging")

%% ESTIMATE GMRF
% Initializing kernels
Q0 = [0 0 0 0 0; 0 0 0 0 0; 0 0 1 0 0; 0 0 0 0 0; 0 0 0 0 0];  % center point
Q1 = [0 0 0 0 0; 0 0 -1 0 0; 0 -1 4 -1 0; 0 0 -1 0 0; 0 0 0 0 0];  % 5-point Laplacian
Q2 = [0 0 1 0 0; 0 2 -8 2 0; 1 -8 20 -8 1; 0 2 -8 2 0; 0 0 1 0 0];

kappa = theta_est.kappa * kFactor;
q = kappa^4 * Q0 + 2 * kappa^2 * Q1 + Q2;
Q = stencil2prec([n,m],q);

tau = 2 * pi / (theta_est.sigma^2);

Q = tau * Q;

figure;
% Plot the precision matrix Q
spy(Q);   % Display the precision matrix as an image
colorbar;     % Display the colorbar to see the scale of values
title('Precision Matrix Q');  % Add a title to the plot
axis equal;   % Ensure that the axes are scaled equally

[qn, qm] = size(Q);
% Shuffle the matrix to randomize the indices (if needed)
row_perm = randperm(qn);  % Random permutation for rows
col_perm = randperm(qm);  % Random permutation for columns
Q_shuffled = Q(row_perm, col_perm);  % Shuffle Q matrix

figure;
% Plot the precision matrix Q
spy(Q_shuffled);   % Display the precision matrix as an image
colorbar;     % Display the colorbar to see the scale of values
title('Precision Matrix Q shuffled');  % Add a title to the plot
axis equal;   % Ensure that the axes are scaled equally

Q_mm = Q(miss_idx, miss_idx);
Q_mo = Q(miss_idx, obs_idx);
Q_oo = Q(obs_idx,obs_idx);

r_m_hat = -Q_mm \ (Q_mo * residuals);
y_m_hat = mu_miss + r_m_hat; 

I_pred = zeros([n*m, 1]);
I_pred(obs_idx) = y_sample;
I_pred(miss_idx) = y_m_hat;


image_pred = reshape(I_pred, n, m);
% I_pred = mat2gray(I_pred);  % Rescale to [0, 1] if necessary



figure;
colormap('gray');

subplot(1,3,1); % 1 row, 3 columns, 1st image
imshow(image);
title('Original (scaled)');

subplot(1,3,2); % 2nd image
imshow(image_obs);
title('Perturbed');

subplot(1,3,3); % 3rd image
imshow(image_pred);
title('Reconstructed');
axis image;
