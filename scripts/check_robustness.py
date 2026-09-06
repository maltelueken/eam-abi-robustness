"""Maximum mean discrepancy between the NPE and MCMC posteriors, per dataset and member.

One row per ensemble member per dataset. For a single-network run there is exactly one
member and the table is what it always was, with a constant `member` column; for an ensemble
the spread of `mmd` across members at a fixed dataset is the inference variability -- how
much of the mismatch is this particular approximator rather than the method -- read on the
same scale as the mismatch itself.
"""

import logging

import bayesflow as bf
import hydra
from omegaconf import DictConfig

from pipeline import (
    accuracy,
    iter_comparable_cases,
    load_case_data,
    load_npe_members,
    num_obs_per_dataset,
    rt_range,
    setup,
)
from results import write_long_csv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_robustness(cfg: DictConfig):
    """Write `robustness/mmd.csv`, one row per ensemble member per dataset per case."""
    artifacts, cases, param_names = setup(cfg)

    records = []

    for case, _pooled_npe, posterior_mcmc, is_converged in iter_comparable_cases(cfg, artifacts, cases, param_names):
        logger.info("Computing MMD for case %s", case.key)

        # Per-dataset properties of the generated data, carried alongside the mismatch: the
        # quantity study 5 selects on, the pair study 6 selects on, and the trial count, which
        # is constant except in study 4, where each subject keeps their own.
        data = load_case_data(case, artifacts)
        rates = accuracy(data["x"], is_converged)
        lowest_rt, highest_rt = rt_range(data["x"], is_converged)
        trial_counts = num_obs_per_dataset(data, is_converged)

        # Members separately, not the pooled posterior `iter_comparable_cases` yields: pooling
        # them first would average the ensemble into one approximator and lose the spread.
        members = load_npe_members(artifacts.npe_samples(case), param_names)[:, is_converged]

        for member, member_draws in enumerate(members):
            for index, (mcmc_draws, npe_draws) in enumerate(zip(posterior_mcmc, member_draws, strict=True)):
                records.append(
                    {
                        **case.labels,
                        "member": member,
                        "dataset": index,
                        "num_obs": int(trial_counts[index]),
                        "accuracy": float(rates[index]),
                        "rt_min": float(lowest_rt[index]),
                        "rt_max": float(highest_rt[index]),
                        "mmd": float(bf.metrics.functional.maximum_mean_discrepancy(mcmc_draws, npe_draws)),
                    },
                )

    path = artifacts.csv("robustness", "mmd")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    check_robustness()
