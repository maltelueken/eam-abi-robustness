"""Posterior median and 95% credible interval per dataset, for NPE and MCMC alike."""

import logging

import hydra
from omegaconf import DictConfig

from metrics import posterior_summary, true_values
from pipeline import accuracy, iter_comparable_cases, load_case_data, load_true_params, rt_range, setup
from results import write_long_csv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_summary_stats(cfg: DictConfig):
    """Write `metrics/summary_stats.csv` in long format."""
    artifacts, cases, param_names = setup(cfg)

    records = []

    for case, posterior_npe, posterior_mcmc, is_converged in iter_comparable_cases(cfg, artifacts, cases, param_names):
        logger.info("Summarizing posteriors for case %s", case.key)

        case_records = list(posterior_summary(posterior_npe, param_names, "npe", case.labels))
        case_records += list(posterior_summary(posterior_mcmc, param_names, "mcmc", case.labels))

        targets = load_true_params(case, artifacts, param_names, is_converged)

        if targets is not None:
            case_records += list(true_values(targets, param_names, case.labels))

        # Properties of each dataset, carried alongside its posterior summaries. Accuracy is
        # study 5's x-axis -- the quantity that study manipulates -- and studies 2 and 4 plot the
        # mismatch against it too, which no script had emitted since the CSVs went long-format.
        # The response-time bounds are the same thing for study 6.
        data_x = load_case_data(case, artifacts)["x"]
        rates = accuracy(data_x, is_converged)
        lowest_rt, highest_rt = rt_range(data_x, is_converged)
        records += [
            {
                **record,
                "accuracy": float(rates[record["dataset"]]),
                "rt_min": float(lowest_rt[record["dataset"]]),
                "rt_max": float(highest_rt[record["dataset"]]),
            }
            for record in case_records
        ]

    path = artifacts.csv("metrics", "summary_stats")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    check_summary_stats()
