"""Sample the trained NPE's posterior for every test case."""

import logging

import hydra
from omegaconf import DictConfig

from data import save_posterior, stack_posterior_members
from ensemble import sample_members
from pipeline import load_case_data, setup
from utils import load_approximator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def predict_npe(cfg: DictConfig):
    """Run posterior sampling per case and save the draws as NetCDF.

    Ensemble members are sampled separately rather than merged into one mixture, and stored
    one per `chain`, so that downstream comparisons can read either the combined posterior or
    a single member's. `test_num_posterior_samples` is drawn *per member*, which keeps each
    member directly comparable to the MCMC reference of the same size.
    """
    approximator, _ = load_approximator(cfg)

    artifacts, cases, param_names = setup(cfg)

    for case in cases:
        logger.info("Loading test data for case %s", case.key)
        conditions = load_case_data(case, artifacts)

        posterior_samples = sample_members(
            approximator,
            conditions,
            num_samples=cfg["test_num_posterior_samples"],
        )

        path = artifacts.npe_samples(case)
        artifacts.ensure(path)
        logger.info("Saving NPE samples for %s member(s) to %s", len(posterior_samples), path)
        save_posterior(path, stack_posterior_members(posterior_samples, param_names), param_names)


if __name__ == "__main__":
    predict_npe()
