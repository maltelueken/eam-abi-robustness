"""Metrics utilities to calculate posterior predictive error statistics."""

import bayesflow.diagnostics.metrics as bf_metrics
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


def posterior_summary(posterior, param_names, method, labels, lower=0.025, upper=0.975):
    """Yield long-format records of posterior median and credible-interval bounds.

    Args:
        posterior: array of shape ``(dataset, draw, param)``.
        param_names: names for the trailing axis.
        method: ``"npe"`` or ``"mcmc"``.
        labels: case-identifying columns to attach to every record.
        lower: lower quantile of the credible interval.
        upper: upper quantile of the credible interval.
    """
    statistics = {
        "median": np.median(posterior, axis=1),
        "lower": np.quantile(posterior, lower, axis=1),
        "upper": np.quantile(posterior, upper, axis=1),
    }

    for statistic, values in statistics.items():
        for dataset_index in range(values.shape[0]):
            for param_index, param in enumerate(param_names):
                yield {
                    **labels,
                    "dataset": dataset_index,
                    "param": param,
                    "method": method,
                    "statistic": statistic,
                    "value": values[dataset_index, param_index],
                }


def true_values(targets, param_names, labels):
    """Yield long-format records of the ground-truth parameter values."""
    for dataset_index in range(targets.shape[0]):
        for param_index, param in enumerate(param_names):
            yield {
                **labels,
                "dataset": dataset_index,
                "param": param,
                "method": "true",
                "statistic": "true",
                "value": targets[dataset_index, param_index],
            }


def recovery_metrics(estimates, targets, param_names, method, labels):
    """Yield long-format records of RMSE, posterior contraction and calibration error.

    These need ground truth, so they are only produced for the simulation studies.
    """
    computed = {
        "rmsd": bf_metrics.root_mean_squared_error,
        "contraction": bf_metrics.posterior_contraction,
        "calibration_error": bf_metrics.calibration_error,
    }

    for statistic, metric_fn in computed.items():
        values = metric_fn(estimates=estimates, targets=targets, variable_names=list(param_names))["values"]

        for param, value in zip(param_names, values, strict=True):
            yield {
                **labels,
                "dataset": pd.NA,
                "param": param,
                "method": method,
                "statistic": statistic,
                "value": value,
            }


def diagnostic_summary(estimates, targets, param_names):
    """Mean RMSE, posterior contraction and calibration error over the given parameters."""
    return tuple(
        metric_fn(estimates=estimates, targets=targets, variable_names=list(param_names))["values"].mean()
        for metric_fn in (
            bf_metrics.root_mean_squared_error,
            bf_metrics.posterior_contraction,
            bf_metrics.calibration_error,
        )
    )
