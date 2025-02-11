import os

from typing import Iterable

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from hydra.utils import instantiate
from matplotlib.lines import Line2D


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


def create_pushforward_plot(sims, param_names=None):
    fig, axarr = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axarr.flat):
        rt = sims["sim_data"][i, :, 0].flatten()
        resp = sims["sim_data"][i, :, 1].flatten()
        params = sims["prior_draws"][i, :]
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
            "Acc: " + str(np.round(resp.mean(), 2)),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        for i, p in enumerate(params):
            if param_names is not None:
                ps = param_names[i] + ": " + str(np.round(p, 2))
            else:
                ps = np.round(p, 2)
            ax.text(
                0.9,
                0.8 - i * 0.1,
                ps,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        # ax.legend(labels=["False", "True"])
        ax.set_ylabel("")
        ax.set_yticks([])
        if i > 19:
            ax.set_xlabel("Simulated RTs (seconds)")
    fig.tight_layout()
    return fig


def create_robustness_2d_plot(
    posterior_draws_npe,
    posterior_draws_mcmc,
    prior_draws=None,
    true_values=None,
    param_names=None,
    height=3,
    label_fontsize=14,
    legend_fontsize=16,
    tick_fontsize=12,
    npe_color="maroon",
    mcmc_color="steelblue",
    prior_color="gold",
    true_color="black",
    post_alpha=0.9,
    prior_alpha=0.7,
):
    _, n_params = posterior_draws_npe.shape

    # Pack posterior draws into a dataframe
    posterior_draws_npe_df = pd.DataFrame(posterior_draws_npe, columns=param_names)

    # Add posterior
    g = sns.PairGrid(posterior_draws_npe_df, height=height)
    g.map_diag(sns.histplot, fill=True, color=npe_color, alpha=post_alpha, kde=True, zorder=0)
    g.map_lower(sns.kdeplot, fill=True, color=npe_color, alpha=post_alpha, zorder=0)

    # Add prior, if given
    if posterior_draws_mcmc is not None:
        posterior_draws_mcmc_df = pd.DataFrame(posterior_draws_mcmc, columns=param_names)
        g.data = posterior_draws_mcmc_df
        g.map_diag(sns.histplot, fill=True, color=mcmc_color, alpha=post_alpha, kde=True, zorder=1)
        g.map_lower(sns.kdeplot, fill=True, color=mcmc_color, alpha=post_alpha, zorder=1)

    # Add prior, if given
    if prior_draws is not None:
        prior_draws_df = pd.DataFrame(prior_draws, columns=param_names)
        g.data = prior_draws_df
        g.map_diag(sns.histplot, fill=True, color=prior_color, alpha=prior_alpha, kde=True, zorder=-1)
        g.map_lower(sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1)

    if true_values is not None:
        true_values_df = pd.DataFrame(np.expand_dims(true_values, 0), columns=param_names)
        g.data = true_values_df
        g.map_diag(sns.rugplot, color=true_color, height=0.1, zorder=2)
        g.map_lower(sns.scatterplot, color=true_color, marker="X", zorder=2)

    # Add legend, if prior also given
    handles = [
        Line2D(xdata=[], ydata=[], color=npe_color, lw=3, alpha=post_alpha),
        Line2D(xdata=[], ydata=[], color=mcmc_color, lw=3, alpha=post_alpha),
        Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha),
    ]
    g.fig.legend(handles, ["Posterior NPE", "Posterior MCMC", "Prior"], fontsize=legend_fontsize, loc="center right")

    # Remove upper axis
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].axis("off")

    # Modify tick sizes
    for i, j in zip(*np.tril_indices_from(g.axes, 1)):
        g.axes[i, j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        g.axes[i, j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Add nice labels
    for i, param_name in enumerate(param_names):
        g.axes[i, 0].set_ylabel(param_name, fontsize=label_fontsize)
        g.axes[len(param_names) - 1, i].set_xlabel(param_name, fontsize=label_fontsize)

    # Add grids
    for i in range(n_params):
        for j in range(n_params):
            g.axes[i, j].grid(alpha=0.5)

    g.tight_layout()

    return g.figure


def create_profile_likelihood_plot(ll_fun, prior_draws, param_names=None, p_range=0.5, num_points=100):
    p_grid = np.linspace(prior_draws - p_range / 2, prior_draws + p_range / 2, num_points)

    x = np.tile(prior_draws, (num_points*len(prior_draws), 1))

    for i in range(len(prior_draws)):
        x[(i*num_points):((i+1)*num_points),i] = p_grid[:, i]

    ll_prior = ll_fun(prior_draws)

    ll = jax.vmap(ll_fun)(x)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))

    for i, ax in enumerate(axes):
        ax.plot(x[(i*num_points):((i+1)*num_points), i], ll[(i*num_points):((i+1)*num_points)])
        ax.plot(prior_draws[i], ll_fun(prior_draws), "o", color="red")
        if param_names is not None:
            ax.set_xlabel(param_names[i])
        ax.set_ylabel("Log-likelihood")

    fig.tight_layout()

    return fig
