"""Script-level helpers shared by the stages in `scripts/`.

The four comparison scripts all begin the same way: for one test case, load the NPE and
MCMC posteriors, drop the datasets whose MCMC chains did not converge, and align the two
on the same parameters and the same number of draws. That is `load_paired_posteriors`.
"""

import logging

import numpy as np
from hydra.utils import instantiate

from artifacts import Artifacts
from config import get_param_names
from data import load_dataset, load_posterior
from mcmc import load_mcmc_posterior
from utils import convert_prior_samples, read_data_from_txt

logger = logging.getLogger(__name__)


def setup(cfg):
    """Return the artifact paths, test cases and parameter names for this run."""
    return Artifacts(cfg), instantiate(cfg["test_case"]), get_param_names(cfg)


def select_cases(cases, key):
    """Filter `cases` down to the one matching `key`, or return all of them when it is None."""
    if key is None:
        return cases

    selected = [case for case in cases if case.key == key]

    if not selected:
        msg = f"No test case named {key!r}; available cases are {[case.key for case in cases]}."
        raise ValueError(msg)

    return selected


def load_case_data(case, artifacts):
    """Load a case's test data as a dict of arrays, from NetCDF or from an empirical file."""
    if case.is_simulated:
        return load_dataset(artifacts.test_data(case))

    return read_data_from_txt(artifacts.test_data(case))


def num_obs_per_dataset(data, is_converged=None):
    """Trial count per dataset, as an integer array of length `num_datasets`.

    A simulated case stores one scalar `num_obs` for the whole batch, an empirical one stores a
    count per subject (`utils.read_data_from_txt`); this is the shape both look like from the
    outside, so callers that only need the number of trials behind a dataset need no branch.

    `is_converged` restricts the result to the datasets whose MCMC chains converged, matching
    the posteriors `load_paired_posteriors` returns.
    """
    counts = np.broadcast_to(
        np.reshape(np.asarray(data["num_obs"]), -1), (np.shape(data["x"])[0],),
    ).astype(int)

    return counts if is_converged is None else counts[is_converged]


def num_obs_groups(data):
    """Split a case's data into blocks of datasets that share one trial count.

    Empirical subjects do not all run for the same number of trials, but the summary network
    and the NUTS driver both want a rectangular batch: `EnsembleApproximator.sample` stacks the
    datasets into one tensor, and `mcmc.fit_mcmc_gpu_batch` vmaps over them, so neither can be
    handed a ragged case in one call. Every stage that consumes a case's data therefore walks
    these blocks and reassembles its output in the original dataset order.

    Returns a list of `(index, block)` pairs, where `index` holds the positions the block's
    datasets occupy in `data` and `block` is `data` with `x` sliced down to that block's own
    trial count -- so the NaN padding `read_data_from_txt` adds never leaves this function.

    A simulated case has a scalar `num_obs` and comes back as a single block holding every
    dataset, unsliced, which is exactly the one call those stages made before.
    """
    num_obs = np.asarray(data["num_obs"])
    num_datasets = np.shape(data["x"])[0]

    if num_obs.ndim == 0:
        return [(np.arange(num_datasets), data)]

    blocks = []

    for value in np.unique(num_obs):
        index = np.flatnonzero(num_obs == value)
        block = {
            key: np.asarray(val)[index] for key, val in data.items()
            if np.shape(val)[:1] == (num_datasets,)
        }
        block["x"] = block["x"][:, : int(value)]
        block["num_obs"] = np.asarray(value)
        blocks.append((index, block))

    return blocks


def restore_dataset_order(blocks, arrays):
    """Concatenate one array per block of `num_obs_groups` back into dataset order.

    Each array's leading axis is the block's datasets, in the order `num_obs_groups` listed
    them; the result's leading axis is every dataset, in the order the case stores them.
    """
    order = np.argsort(np.concatenate([index for index, _ in blocks]))

    return np.concatenate(list(arrays), axis=0)[order]


def load_npe_posterior(filename, param_names):
    """Load NPE samples as `(dataset, draw, param)`, selected by parameter name."""
    posterior = load_posterior(filename)

    return (
        posterior["theta"]
        .sel(param=list(param_names))
        .stack(sample=("chain", "draw"))
        .transpose("dataset", "sample", "param")
        .to_numpy()
    )


def load_npe_members(filename, param_names):
    """Load NPE samples per ensemble member, as `(member, dataset, draw, param)`.

    The counterpart of :func:`load_npe_posterior`, which pools the members into the
    ensemble's combined posterior. Keep them apart when the question is how much the
    approximation itself varies: `data.stack_posterior_members` writes one chain per member,
    and a single-network run is the one-member case, so this works for both.
    """
    posterior = load_posterior(filename)

    return (
        posterior["theta"]
        .sel(param=list(param_names))
        .transpose("chain", "dataset", "draw", "param")
        .to_numpy()
    )


def load_paired_posteriors(cfg, artifacts, case, param_names):
    """Load a case's NPE and MCMC posteriors, aligned and restricted to converged datasets.

    Returns `(posterior_npe, posterior_mcmc, is_converged)`, both posteriors shaped
    `(num_converged_datasets, num_draws, len(param_names))`.
    """
    logger.info("Loading MCMC samples from %s", artifacts.mcmc_samples(case))
    posterior_mcmc, is_converged = load_mcmc_posterior(
        artifacts.mcmc_samples(case),
        to_constrained=instantiate(cfg["mcmc_to_constrained"]),
        param_names=param_names,
        psrf_threshold=cfg["psrf_threshold"],
        num_target_samples=cfg["test_num_posterior_samples"],
    )

    logger.info("Loading NPE samples from %s", artifacts.npe_samples(case))
    posterior_npe = load_npe_posterior(artifacts.npe_samples(case), param_names)[is_converged]

    return posterior_npe, posterior_mcmc, is_converged


def iter_comparable_cases(cfg, artifacts, cases, param_names):
    """Yield `(case, posterior_npe, posterior_mcmc, is_converged)` for every usable case.

    A case whose MCMC chains all failed to converge has no ground truth to compare against,
    so it is skipped with a warning rather than aborting the whole run -- one bad case in a
    sweep should not cost the other twenty-odd.
    """
    for case in cases:
        posterior_npe, posterior_mcmc, is_converged = load_paired_posteriors(cfg, artifacts, case, param_names)

        if not is_converged.any():
            logger.warning(
                "Skipping case %s: none of its %s MCMC fits reached R-hat < %s. "
                "Consider more warmup steps (mcmc_sampling_fun.num_steps_warmup) or a looser psrf_threshold.",
                case.key,
                is_converged.size,
                cfg["psrf_threshold"],
            )
            continue

        yield case, posterior_npe, posterior_mcmc, is_converged


def load_true_params(case, artifacts, param_names, is_converged):
    """Load the ground-truth parameters for a simulated case, restricted to converged datasets.

    Returns None for empirical cases, which have no ground truth.
    """
    if not case.is_simulated:
        return None

    forward_dict = load_dataset(artifacts.test_data(case))

    return convert_prior_samples(forward_dict, param_names)[is_converged]


def accuracy(data_x, is_converged=None):
    """Mean accuracy per dataset -- the fraction of trials the correct accumulator won.

    Channel 1 of `x` is 1 when the true accumulator finished first, so this is accuracy, not the
    error rate the previous name claimed. Study 5 manipulates exactly this quantity, so the two
    had better not be confused.

    `is_converged` restricts the result to the datasets whose MCMC chains converged, matching the
    posteriors `load_paired_posteriors` returns.
    """
    per_dataset = np.nanmean(np.asarray(data_x)[:, :, 1], axis=1)

    return per_dataset if is_converged is None else per_dataset[is_converged]


def rt_range(data_x, is_converged=None):
    """Fastest and slowest response time per dataset.

    Channel 0 of `x` is the response time. Study 6 selects on exactly this pair -- it keeps only
    datasets whose response times all fall in a window -- so the figures need each dataset's
    realized bounds next to the window it was drawn from, the way study 5 needs `accuracy`. The
    windows are nested, so the realized slowest response time is the only continuous axis the
    test cases leave.

    `is_converged` restricts the result to the datasets whose MCMC chains converged, matching the
    posteriors `load_paired_posteriors` returns.
    """
    rt = np.asarray(data_x)[:, :, 0]
    lowest, highest = np.nanmin(rt, axis=1), np.nanmax(rt, axis=1)

    if is_converged is None:
        return lowest, highest

    return lowest[is_converged], highest[is_converged]
