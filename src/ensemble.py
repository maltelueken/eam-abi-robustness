"""Ensembles of neural posterior estimators.

A single NPE gives one approximation of the posterior; nothing in its output says how much
of the deviation from the MCMC reference is *this* approximator rather than the method. An
ensemble answers that: train several members that differ only in their random
initialisation and their SGD path, sample each one separately, and the spread across
members is the inference variability, measurable on the same footing as the NPE-vs-MCMC
mismatch the rest of the pipeline reports.

Two choices make that reading valid, and both are guarded in `tests/test_ensemble.py`:

* **Members share their training data exactly** (`data_reuse=1.0`, the default of
  `bayesflow.datasets.EnsembleDataset`, which `EnsembleApproximator.build_dataset` builds).
  Every member sees the same batches in the same order, so what is left between them is
  initialisation and optimisation, not a different sample of the training prior. Lower
  values would fold simulation noise into the spread and make it a bagging estimate
  instead.
* **Members share their architecture but not their weights.** Every member is instantiated
  from the *same* config nodes, so the architecture the Optuna sweep selected applies to
  all of them; each instantiation is separate, so no tensor is shared. Sharing an instance
  -- what `EnsembleWorkflow`'s size mode does with a summary network -- would both collapse
  the spread and break serialisation, which is why this module goes through the workflow's
  dictionary mode instead.

Members are trained jointly, in one Keras model, but not *coupled*: `EnsembleApproximator`
sums the member losses, and the gradient of that sum with respect to one member's variables
is that member's own gradient. Adam keeps per-variable moments, and the configured
`clipnorm` clips each gradient tensor on its own (`global_clipnorm` would not), so one
training run is equivalent to `ensemble_size` independent ones on the same data stream --
at `ensemble_size` times the cost per step.

The factory returns the `EnsembleApproximator` rather than the workflow that built it, so
`scripts/train_npe.py` and `scripts/predict_npe.py` treat it exactly like the single-network
`ContinuousApproximator` -- compile, `fit(simulator=...)`, checkpoint, `keras.saving.load_model`.
"""

import bayesflow as bf
import numpy as np
from hydra.utils import instantiate

# Below two members `EnsembleOnlineDataset` refuses to build, and the spread this module
# exists to measure is undefined anyway.
MIN_ENSEMBLE_SIZE = 2


def member_names(ensemble_size):
    """Names of the ensemble members, in the order they are stored along `chain`.

    `save_posterior` writes one chain per member, so the position of a name in this list is
    the chain index its draws end up at.
    """
    return [str(index) for index in range(ensemble_size)]


def create_ensemble_approximator(
    adapter,
    inference_network,
    summary_network,
    ensemble_size,
    **kwargs,
):
    """Build an `EnsembleApproximator` of `ensemble_size` identically configured members.

    Instantiated with `_recursive_: false` (see `conf/approximator/ensemble_approximator.yaml`)
    so that the network arguments arrive as config nodes rather than as objects: each member
    needs its *own* networks, and the only way to get independently initialised ones that are
    still guaranteed to match the tuned architecture is to instantiate the same node once per
    member.
    """
    if ensemble_size < MIN_ENSEMBLE_SIZE:
        msg = f"An ensemble needs at least two members; got ensemble_size={ensemble_size}."
        raise ValueError(msg)

    names = member_names(ensemble_size)

    workflow = bf.workflows.EnsembleWorkflow(
        adapter=instantiate(adapter, _convert_="partial"),
        inference_networks={name: instantiate(inference_network, _convert_="partial") for name in names},
        summary_networks={name: instantiate(summary_network, _convert_="partial") for name in names},
        **kwargs,
    )

    return workflow.approximator


def num_members(approximator):
    """How many members `approximator` holds -- one for a plain `ContinuousApproximator`.

    The same one-member reading `sample_members` takes, so that a single-network run reports
    its cost on the same scale as an ensemble's: `scripts/train_npe.py` records this alongside
    the training time, and the training of an ensemble buys this many networks for it.
    """
    if isinstance(approximator, bf.approximators.EnsembleApproximator):
        return len(approximator.approximators)

    return 1


def sample_members(approximator, conditions, num_samples):
    """Draw `num_samples` per ensemble member, as `{member: {param: (dataset, draw, 1)}}`.

    A plain `ContinuousApproximator` is reported as a one-member ensemble, so that
    `scripts/predict_npe.py` needs no branch and a single-network run keeps writing exactly
    the one-chain posterior it wrote before.

    `merge_members=False` is what makes this an ensemble study rather than a bigger NPE: the
    default merges the members into one mixture, which is the ensemble's *combined* posterior
    and averages away precisely the variability being measured.
    """
    if isinstance(approximator, bf.approximators.EnsembleApproximator):
        return approximator.sample(
            conditions=conditions,
            num_samples=num_samples,
            merge_members=False,
        )

    return {"0": approximator.sample(conditions=conditions, num_samples=num_samples)}


def sample_members_by_group(approximator, blocks, num_samples):
    """`sample_members` over the blocks of `pipeline.num_obs_groups`, in dataset order.

    An empirical case's subjects no longer share a trial count, so its datasets cannot be
    stacked into the one tensor `sample` wants. Each block is sampled on its own and the draws
    are put back in the case's own dataset order, so what `save_posterior` writes is indexed
    the same way the test data and the MCMC fits are.

    A simulated case is a single block and comes out of here bit-for-bit as it went in.
    """
    per_block = [sample_members(approximator, block, num_samples) for _, block in blocks]

    return _merge_blocks(blocks, per_block, nested=True)


def summarize_members(approximator, conditions):
    """Each member's learned summary statistics for `conditions`, as `{member: (dataset, feature)}`.

    The counterpart of `sample_members` on the other side of the network: where that reads what a
    member infers, this reads what it *sees*. `scripts/prior_distance.py` compares training and
    test data here rather than in a hand-picked statistic, because this is the representation the
    inference network is actually conditioned on -- two datasets that the summary network maps to
    the same place are indistinguishable to the NPE no matter how far apart their raw parameters
    were, and two that it separates are a gap no amount of inference-network capacity can close.

    Members are kept apart for the same reason `sample_members` keeps them apart: they are
    separately initialised summary networks, and the spread across them is the variability
    attributable to this particular approximator rather than to the method.
    """
    if isinstance(approximator, bf.approximators.EnsembleApproximator):
        return {
            name: np.asarray(member.summarize(conditions))
            for name, member in approximator.approximators.items()
        }

    return {"0": np.asarray(approximator.summarize(conditions))}


def summarize_members_by_group(approximator, blocks):
    """`summarize_members` over the blocks of `pipeline.num_obs_groups`, in dataset order.

    The counterpart of `sample_members_by_group` on the other side of the network, and needed
    for the same reason: the summary network takes one rectangular batch, and an empirical
    case's subjects do not all have the same number of trials.
    """
    per_block = [summarize_members(approximator, block) for _, block in blocks]

    return _merge_blocks(blocks, per_block)


def _merge_blocks(blocks, per_block, nested=False):
    """Reassemble one `{member: ...}` result per block into a single one in dataset order.

    `nested` says whether a member's value is an array (`summarize_members`) or a
    `{param: array}` dict (`sample_members`).

    The order restore is duplicated from `pipeline.restore_dataset_order` rather than imported:
    `pipeline` pulls in `mcmc`, which flips JAX into float64 at import, and this module is what
    `train_npe` instantiates the approximator through.
    """
    order = np.argsort(np.concatenate([index for index, _ in blocks]))

    def merged(arrays):
        return np.concatenate(list(arrays), axis=0)[order]

    merged_members = {}

    for member in per_block[0]:
        if nested:
            merged_members[member] = {
                param: merged(result[member][param] for result in per_block)
                for param in per_block[0][member]
            }
        else:
            merged_members[member] = merged(result[member] for result in per_block)

    return merged_members
