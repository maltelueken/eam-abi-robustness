"""How far each test case is from what the approximator was trained on.

The `check_*` scripts measure how badly the NPE misses; this measures how far the case asked it
to go. Those are not the same axis: study 2's `drift_slope_loc` is a truncated-normal location, so
equal steps along the sweep are not equal steps in distribution, and the sweep's two ends differ
by roughly an order of magnitude in divergence while looking symmetric on the axis.

Two spaces, because they answer different questions:

* `parameter` -- `KL(test prior || training prior)`, from the priors' *analytic* densities
  (`mcmc_log_prior_fun`) rather than estimated from samples, so the only approximation is the
  Monte Carlo average of a log ratio known exactly at every draw. One number per case: the priors
  factorize and only the swept factor differs, so the joint KL *is* that factor's KL and a
  per-parameter breakdown would be zeros by construction. For a hierarchical model the comparison
  is against the mixture over the range its meta simulator randomizes, not against a point prior.
  KL is an expectation under the *test* prior, so this needs draws from that prior only, and it
  needs the parameters rather than the datasets -- it goes to `prior_simulator` directly and never
  simulates a trial.

* `summary` -- MMD between the training data and the case's data **in the summary network's
  output space**, one row per ensemble member. This is the representation the inference network is
  actually conditioned on: two datasets the summary network maps to the same place are
  indistinguishable to the NPE however far apart their parameters were, and two it separates are a
  gap no amount of inference-network capacity can close. Reading it in the network's space rather
  than in hand-picked statistics is what makes it a statement about *this* approximator -- which
  is why it is per member, like `check_robustness.py`'s mismatch.

Which cases get which:

* Studies 2 and 3 move a prior hyperparameter, and get both.
* Studies 5 and 6 get only the summary-space row. The prior they *sample* is unchanged -- the band
  rejects whole datasets afterwards -- so there is no shifted density to evaluate. The effective
  prior among accepted draws does move, but only implicitly, which is exactly why those studies
  are framed as amortization coverage rather than prior misspecification.
* Study 4 gets only the summary-space row too, and it is the interesting one: it compares the
  *empirical* subject data against the simulations the network was trained on, which is the
  amortization gap that study exists to probe. Its comparison sample is drawn at the subject
  file's own trial count -- the summary network takes a set, but its embedding still moves with
  the set size, and a mismatch there would be read as a distance.

Runs after `train_npe`, since it needs the trained summary networks.
"""

import logging

import bayesflow as bf
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from divergence import kl_from_log_densities, mixture_log_density, standardize
from ensemble import summarize_members
from pipeline import load_case_data, setup
from results import write_long_csv
from utils import load_approximator

logger = logging.getLogger(__name__)

NOT_HYPERPARAMETERS = {"_target_", "_partial_", "rng"}


def _parameter_matrix(prior_dict, param_names):
    """Stack prior draws into `(draw, parameter)` in the order the NPE infers them."""
    return np.column_stack([np.reshape(prior_dict[name], -1) for name in param_names])


def _prior_hyperparameters(cfg):
    """Names of the prior's hyperparameters, as its yaml binds them.

    Used to tell a case that moves the prior (studies 2 and 3, whose label names one of these)
    from one that moves the data (studies 5 and 6, whose labels are band edges) or supplies it
    outright (study 4, whose label is a subject file).
    """
    return set(cfg["simulator"]["prior_simulator"]["sample_fn"]) - NOT_HYPERPARAMETERS


def _randomized_hyperparameter(cfg):
    """The hyperparameter a hierarchical model is amortized over, and its range, or `None`."""
    meta = cfg["simulator"].get("meta_simulator")

    if meta is None:
        return None

    return {
        "name": meta["sample_fn"]["name"][0],
        "lower": float(meta["sample_fn"]["min_value"][0]),
        "upper": float(meta["sample_fn"]["max_value"][0]),
    }


def _training_summaries(simulator, approximator, num_draws, num_obs):
    """Embed a fresh sample of the training distribution, at a given trial count."""
    trained = simulator.sample(num_draws, num_obs=np.array(num_obs))

    return summarize_members(approximator, trained)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def prior_distance(cfg: DictConfig):
    """Write `metrics/prior_distance.csv` in long format."""
    approximator, simulator = load_approximator(cfg)

    artifacts, cases, param_names = setup(cfg)
    log_prior = instantiate(cfg["mcmc_log_prior_fun"])
    hyperparameters = _prior_hyperparameters(cfg)
    randomized = _randomized_hyperparameter(cfg)
    num_draws = cfg["prior_distance_num_draws"]
    num_mmd_draws = cfg["prior_distance_mmd_draws"]

    logger.info("Embedding %s training datasets", num_mmd_draws)
    trained_summaries = _training_summaries(simulator, approximator, num_mmd_draws, cfg["test_num_obs"])

    records = []

    for case in cases:
        logger.info("Measuring distance for case %s", case.key)

        swept = {name: float(value) for name, value in case.labels.items() if name in hyperparameters}

        if swept:
            # Parameters only, and from the test prior alone: KL is an expectation under it, and
            # nothing here needs a simulated trial. Cheap enough to afford many more draws than
            # the MMD below, which is what keeps the Monte Carlo error negligible.
            draws = _parameter_matrix(
                simulator.prior_simulator.sample((num_draws,), **case.sim_kwargs), param_names,
            )
            log_density_test = np.asarray(log_prior(draws, **swept), dtype=float)
            log_density_trained = (
                mixture_log_density(log_prior, draws, **randomized)
                if randomized is not None
                else np.asarray(log_prior(draws), dtype=float)
            )
            records.append({
                **case.labels,
                "member": "",
                "space": "parameter",
                "statistic": "kl",
                "value": kl_from_log_densities(log_density_test, log_density_trained),
            })

        if case.is_simulated:
            tested = simulator.sample(num_mmd_draws, **case.sim_kwargs)
            reference_summaries = trained_summaries
        else:
            # Empirical data: compare them against simulations of the same length, since the
            # summary network's embedding moves with the size of the set it is given.
            tested = load_case_data(case, artifacts)
            reference_summaries = _training_summaries(
                simulator, approximator, num_mmd_draws, tested["num_obs"],
            )
            logger.info(
                "Case %s is empirical: %s datasets of %s trials against %s simulated ones",
                case.key, np.shape(tested["x"])[0], tested["num_obs"], num_mmd_draws,
            )

        tested_summaries = summarize_members(approximator, tested)

        records += [
            {
                **case.labels,
                "member": member,
                "space": "summary",
                "statistic": "mmd",
                "value": float(bf.metrics.functional.maximum_mean_discrepancy(*standardize(
                    reference_summary, reference_summary, tested_summaries[member],
                ))),
            }
            for member, reference_summary in reference_summaries.items()
        ]

    path = artifacts.csv("metrics", "prior_distance")
    logger.info("Writing %s", path)
    write_long_csv(path, records)


if __name__ == "__main__":
    prior_distance()
