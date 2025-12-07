"""General utility functions."""

import os

import keras
import numpy as np
import polars as pl

from hydra.utils import instantiate


def create_missing_dirs(dir_list):
    """Create a list of directories."""
    for dirname in dir_list:
        if not (os.path.exists(dirname)):
            os.makedirs(dirname, exist_ok=True)


def convert_prior_samples(forward_dict, param_names):
    """Reorder dimensions of prior samples."""
    return np.moveaxis(
        np.array([forward_dict[key] for key in param_names]).squeeze(), [0, 1], [1, 0]
    )


def convert_posterior_samples(forward_dict, param_names):
    """Reorder dimensions of posterior samples."""
    return np.moveaxis(
        np.array([forward_dict[key] for key in param_names]).squeeze(),
        [0, 1, 2],
        [2, 0, 1],
    )


def get_decay_steps(num_epochs, num_batches):
    """Calculate steps for cosine decay."""
    return num_epochs * num_batches


def load_approximator(cfg):
    """Load an approximator object."""
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    approximator = keras.saving.load_model(cfg["callbacks"][1]["filepath"])

    return approximator, simulator


def trim_num_obs(x, min_num_obs):
    """Trim an array."""
    return x[:min_num_obs]


def read_data_from_txt(filename, trim=True):
    """Read and preprocess an empirical dataset from a text file."""
    df = pl.read_csv(filename, separator=" ").filter(
        pl.col("block").eq("test") & pl.col("trl_number").gt(0)
    )

    if trim:
        df = (
            df.with_columns(pl.col("RT").log().alias("log_RT")).with_columns(
                (
                    pl.col("log_RT").quantile(0.75) - pl.col("log_RT").quantile(0.25)
                ).alias("IQR")
            )
            # Remove trials with log(RT) outside of 3 x IQR from first or third quartile
            .filter(
                pl.col("log_RT").le(pl.col("log_RT").quantile(0.75) + 3 * pl.col("IQR"))
                & pl.col("log_RT").ge(
                    pl.col("log_RT").quantile(0.25) - 3 * pl.col("IQR")
                )
            )
        )

    min_trials = (
        df.group_by("pp").agg(pl.len().alias("n")).select(pl.col("n").min()).item()
    )

    df = (
        df.group_by("pp")
        .agg(
            pl.col("RT").mean().alias("mean_RT"),
            pl.col("acc").mean().alias("mean_acc"),
            pl.col("acc"),
            pl.col("RT") / 1_000,
        )
        .with_columns(
            (pl.col("mean_RT").quantile(0.75) - pl.col("mean_RT").quantile(0.25)).alias(
                "mean_RT_IQR"
            ),
            (
                pl.col("mean_acc").quantile(0.75) - pl.col("mean_acc").quantile(0.25)
            ).alias("mean_acc_IQR"),
        )
        # Remove subjects with mean RT or accuracy outside of 3 x IQR from first or third quartile
        .filter(
            pl.col("mean_RT").le(
                pl.col("mean_RT").quantile(0.75) + 3 * pl.col("mean_RT_IQR")
            ),
            pl.col("mean_RT").ge(
                pl.col("mean_RT").quantile(0.25) - 3 * pl.col("mean_RT_IQR")
            ),
            pl.col("mean_acc").le(
                pl.col("mean_acc").quantile(0.75) + 3 * pl.col("mean_acc_IQR")
            ),
            pl.col("mean_acc").ge(
                pl.col("mean_acc").quantile(0.25) - 3 * pl.col("mean_acc_IQR")
            ),
        )
        .sort("pp")  # Sort so that order is always the same
    )

    x = df.select(pl.col("RT"), pl.col("acc")).to_numpy()

    x = np.array(
        [
            np.array([trim_num_obs(e[0], min_trials), trim_num_obs(e[1], min_trials)]).T
            for e in x
        ]
    )

    # Return list with dicts containing data for a single subject as single batch
    return dict(x=x, num_obs=min_trials)
