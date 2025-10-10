
import numpy as np
import pandas as pd

def calibration_error(
    samples,
    targets,
    min_quantile=0.05,
    max_quantile=0.95,
    resolution=20
):
    # Define alpha values and the corresponding quantile bounds
    alphas = np.linspace(start=min_quantile, stop=max_quantile, num=resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    # Compute quantiles for each alpha, for each dataset and parameter
    quantiles = np.quantile(samples, [lowers, uppers], axis=1)

    # Shape: (2, resolution, num_datasets, num_params)
    lower_bounds, upper_bounds = quantiles[0], quantiles[1]

    # Compute masks for inliers
    lower_mask = lower_bounds <= targets[None, ...]
    upper_mask = upper_bounds >= targets[None, ...]

    # Logical AND to identify inliers for each alpha
    inlier_id = np.logical_and(lower_mask, upper_mask)

    # Compute the relative number of inliers for each alpha
    alpha_pred = np.mean(inlier_id, axis=1)

    # Calculate absolute error between predicted inliers and alpha
    absolute_errors = np.abs(alpha_pred - alphas[:, None])

    return absolute_errors


def calc_posterior_predictive(data, posterior, num_obs, simulator, num_samples):
    idx = []
    posterior_sample = []
    acc_true = []
    acc_est = []
    quantile = []
    rt_true = []
    rt_est = []

    for i in range(data.shape[0]):
        for j in range(num_samples):
            sim = simulator.experiment_simulator.sample_fn(
                v_intercept = posterior[i,j,0],
                v_slope= posterior[i,j,1],
                s_true = posterior[i,j,2],
                b = posterior[i,j,3],
                t0 = posterior[i,j,4],
                num_obs=num_obs
            )
            idx.append(i)
            posterior_sample.append(j)
            acc_est.append(sim["x"][:,1].mean())
            acc_true.append(data[i,:,1].mean())
            quantile.append((np.arange(1, 10)/10).tolist())
            rt_est.append(np.quantile(sim["x"][:,0], q=np.arange(1, 10)/10).tolist())
            rt_true.append(np.quantile(data[i,:,0], q=np.arange(1, 10)/10).tolist())

    return pd.DataFrame({
        "id": idx,
        "sample": posterior_sample,
        "acc_true": acc_true,
        "acc_est": acc_est,
        "quantile": quantile,
        "rt_true": rt_true,
        "rt_est": rt_est
    }).explode(["quantile", "rt_true", "rt_est"])
