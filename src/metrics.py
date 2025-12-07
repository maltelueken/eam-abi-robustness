
import numpy as np
import pandas as pd


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
