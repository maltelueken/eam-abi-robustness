import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"
import logging

import bayesflow as bf
import bayesflow.diagnostics.metrics as bf_metrics
import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_npe(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"], _convert_="partial")
    approximator = instantiate(cfg["approximator"], _convert_="partial")
    optimizer = instantiate(cfg["optimizer"], _convert_="partial")

    approximator.compile(optimizer)

    callbacks = instantiate(cfg["callbacks"], _convert_="partial") # [keras.callbacks.TensorBoard()]

    history = approximator.fit(
        epochs=cfg["epochs"],
        num_batches=cfg["iterations_per_epoch"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        simulator=simulator
    )

    _ = bf.diagnostics.plots.loss(history)

    plt.savefig("loss_history.png")

    diag_sample = simulator.sample(
        cfg["diag_batch_size"], num_obs=np.array(cfg["eval_num_obs"])
    )

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    conditions = {k: v for k, v in diag_sample.items() if k not in param_names}

    posterior_samples = approximator.sample(
        num_samples=cfg["diag_num_posterior_samples"], 
        conditions=conditions
    )

    root_mean_squared_error = bf_metrics.root_mean_squared_error(
        estimates=posterior_samples,
        targets=diag_sample,
        variable_names=param_names
    )

    contraction = bf_metrics.posterior_contraction(
        estimates=posterior_samples,
        targets=diag_sample,
        variable_names=param_names
    )

    calibration_errors = bf_metrics.calibration_error(
        estimates=posterior_samples,
        targets=diag_sample,
        variable_names=param_names
    )

    return root_mean_squared_error["values"].mean(), contraction["values"].mean(), calibration_errors["values"].mean()


if __name__ == "__main__":
    train_npe()
