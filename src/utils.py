import os

from typing import Iterable

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

from hydra.utils import instantiate


def create_missing_dirs(dir_list):
    for dirname in dir_list:
        if not (os.path.exists(dirname)):
            os.mkdir(dirname)


def get_default_rng():
    return np.random.default_rng(2024)


def check_rt_dist(x):
    return 0.4 <= x.mean() <= 2.5 and 0.4 <= np.median(x) <= 2.5 and 0.1 <= stats.iqr(x) <= 2.0 and x.max() >= 1.5 and x.min() <= 0.5


def check_mcmc_trace(trace):
    return np.all(az.rhat(trace).to_dataarray().to_numpy() < 1.01) and np.all(az.ess(trace).to_dataarray().to_numpy() >= 1000)


def sub_instantiate(cfg):
    return {k: instantiate(v) if isinstance(v, Iterable) and "_target_" in v else v for k, v in cfg.items()}


def create_pushforward_plot(sims, check_dist=False):
    fig, axarr = plt.subplots(2, 5, figsize=(12, 4))
    for i, ax in enumerate(axarr.flat):
        rt = sims["sim_data"][i, :, 0].flatten()
        resp = sims["sim_data"][i, :, 1].flatten()
        sns.histplot(
            rt[resp == 1], color="darkgreen", alpha=0.5, ax=ax
        )
        sns.histplot(
            rt[resp == 0], color="maroon", alpha=0.5, ax=ax
        )
        sns.despine(ax=ax)
        ax.vlines([rt[resp == 1].mean(), rt[resp == 0].mean()], ymin = 0, ymax = 1, color = ["darkgreen", "maroon"], linestyle = '-', 
           transform=ax.get_xaxis_transform())
        ax.text(
            0.9,
            0.9,
            np.round(resp.mean(), 2),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        # ax.legend(labels=["False", "True"])
        ax.set_ylabel("")
        ax.set_yticks([])
        if i > 4:
            ax.set_xlabel("Simulated RTs (seconds)")
    fig.tight_layout()
    return fig
