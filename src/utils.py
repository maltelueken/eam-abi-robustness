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
        sns.histplot(
            rt, color="maroon", alpha=0.75, ax=ax
        )
        sns.despine(ax=ax)
        ax.text(
            0.9,
            0.9,
            np.round(sims["sim_data"][i, :, 1].mean(), 2),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        if check_dist:
            valid_rt_dist = check_rt_dist(rt)
            ax.text(0.7, 0.9, valid_rt_dist, horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,)
        ax.set_ylabel("")
        ax.set_yticks([])
        if i > 4:
            ax.set_xlabel("Simulated RTs (seconds)")
    fig.tight_layout()
    return fig
