"""Metrics utilities to calculate posterior predictive error statistics."""

import numpy as np
import pandas as pd


def calc_posterior_predictive(data, posterior, num_obs, simulator, num_samples, param_names):
    """Calculate posterior predictive statistics for response time and accuracy.

    `param_names` must list the model's parameters in the same order as `posterior`'s last
    axis -- i.e. the adapter's `inference_variables` -- so that this works for any model
    family (the RDM's five parameters, the LBA's six) rather than a fixed set.
    """
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
                **dict(zip(param_names, posterior[i, j])),
                num_obs=num_obs,
            )
            idx.append(i)
            posterior_sample.append(j)
            acc_est.append(sim["x"][:, 1].mean())
            acc_true.append(data[i, :, 1].mean())
            quantile.append((np.arange(1, 10) / 10).tolist())
            rt_est.append(np.quantile(sim["x"][:, 0], q=np.arange(1, 10) / 10).tolist())
            rt_true.append(np.quantile(data[i, :, 0], q=np.arange(1, 10) / 10).tolist())

    return pd.DataFrame(
        {
            "id": idx,
            "sample": posterior_sample,
            "acc_true": acc_true,
            "acc_est": acc_est,
            "quantile": quantile,
            "rt_true": rt_true,
            "rt_est": rt_est,
        },
    ).explode(["quantile", "rt_true", "rt_est"])
