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
