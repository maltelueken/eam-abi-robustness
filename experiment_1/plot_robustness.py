
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import create_missing_dirs

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def plot_robustness(cfg: DictConfig):
    sample_sizes = instantiate(cfg["diag_num_obs"])

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    create_missing_dirs(["robustness"])

    df_robustness = pd.read_csv(os.path.join("robustness", "mmd.csv"))

    _ = sns.boxplot(df_robustness, x="sample_size", y="mmd")

    plt.savefig(os.path.join("robustness", "mmd.png"))

if __name__ == "__main__":
    plot_robustness()