"""Data utilities for saving and loading simulated data and posterior samples as NetCDF.

Two kinds of artifact live here:

* **Datasets** -- simulator output (prior draws, design variables and the simulated
  response-time/choice array ``x``). Stored as a plain :class:`xarray.Dataset`.
* **Posteriors** -- NPE and MCMC posterior samples alike. Stored as an ArviZ-style
  :class:`xarray.DataTree` with a ``posterior`` group holding a single ``theta`` variable
  with dims ``(chain, draw, dataset, param)``. A single-network NPE simply has ``chain`` of
  size 1; an ensemble stores one chain per member (see :func:`stack_posterior_members`).

Storing both posterior kinds in the same layout means one reader serves both, and naming
the ``param`` axis means parameters are selected by name rather than by position -- which
matters because the hierarchical ("meta") models store a prior hyperparameter ahead of the
subject-level ones that the NPE infers.

The posterior half of this module is a two-line adapter over :mod:`eamax.io`, which owns
both schemas and, deliberately, does nothing but read and write -- no diagnostic, no
threshold, no thinning. Those live in :func:`mcmc.load_mcmc_posterior`, where they are
visible. The dataset half stays here: simulator output is BayesFlow's shape-sensitive dict,
which is this study's concern rather than `eamax`'s.

``load_hdf5`` is retained as a read-only shim for the HDF5 archives produced by earlier
versions of the pipeline; see ``scripts/convert_hdf5_to_netcdf.py``.
"""

import h5py
import numpy as np
import xarray as xr
from eamax.io import open_dataset_posterior, save_dataset_posterior

# Dimension names for simulator output. Anything not listed falls back to the positional
# defaults below, which is enough for the (batch, 1) parameter draws.
DATASET_DIMS = {
    "x": ("dataset", "obs", "channel"),
}
DEFAULT_DATASET_DIMS = ("dataset", "component", "extra")


def _dims_for(key, ndim):
    """Return dimension names for a simulator-output entry of the given rank."""
    dims = DATASET_DIMS.get(key)

    if dims is not None:
        if len(dims) != ndim:
            msg = f"Entry {key!r} has rank {ndim}, but {dims} declares {len(dims)} dimensions."
            raise ValueError(msg)
        return dims

    if ndim > len(DEFAULT_DATASET_DIMS):
        msg = f"Entry {key!r} has rank {ndim}, which exceeds the {len(DEFAULT_DATASET_DIMS)} default dimensions."
        raise ValueError(msg)

    return DEFAULT_DATASET_DIMS[:ndim]


def save_dataset(filename, data_dict):
    """Save a simulator-output dictionary to a NetCDF file.

    Array shapes round-trip exactly, including trailing singleton axes, because BayesFlow's
    adapter is shape-sensitive.
    """
    data_vars = {}

    for key, val in data_dict.items():
        if val is None:
            continue
        arr = np.asarray(val)
        data_vars[key] = (_dims_for(key, arr.ndim), arr)

    xr.Dataset(data_vars).to_netcdf(filename)


def load_dataset(filename):
    """Load a simulator-output dictionary from a NetCDF file.

    Returns plain numpy arrays rather than an :class:`xarray.Dataset`, since this feeds
    straight into BayesFlow, which expects a dict of arrays.
    """
    with xr.open_dataset(filename) as dataset:
        return {key: val.to_numpy() for key, val in dataset.items()}


def save_posterior(filename, theta, param_names):
    """Save posterior samples already shaped ``(chain, draw, dataset, param)`` to a NetCDF file.

    ``layout="arviz"`` because NPE draws arrive in that order already -- they are independent,
    so a member becomes a chain and there is nothing to transpose.
    :func:`mcmc.save_mcmc_posterior` writes the sampler's own ``(dataset, draw, chain, param)``
    through the same function under ``layout="sampler"``.
    """
    save_dataset_posterior(filename, theta, param_names, layout="arviz")


def load_posterior(filename):
    """Load posterior samples saved by :func:`save_posterior`.

    Returns the ``posterior`` group as an :class:`xarray.Dataset`, so that ``theta`` can be
    selected by parameter name and thinned along ``draw``. :func:`mcmc.load_mcmc_posterior`
    goes through :func:`eamax.io.load_dataset_posterior` instead, which back-transforms and
    selects on the way out; this one is for readers that want the labelled array.
    """
    return open_dataset_posterior(filename)


def stack_posterior_dict(samples, param_names):
    """Stack a BayesFlow ``{param: (dataset, draw, 1)}`` dict into ``(chain, draw, dataset, param)``.

    NPE draws are independent, so they are stored as a single chain.
    """
    columns = [np.asarray(samples[name]).squeeze(axis=-1) for name in param_names]

    # (dataset, draw, param) -> (chain=1, draw, dataset, param)
    theta = np.stack(columns, axis=-1)

    return np.transpose(theta, (1, 0, 2))[np.newaxis, ...]


def stack_posterior_members(samples_by_member, param_names):
    """Stack one BayesFlow sample dict per ensemble member into ``(chain, draw, dataset, param)``.

    Each member becomes one chain, in the iteration order of ``samples_by_member`` -- which is
    the member order of :func:`ensemble.member_names`. Reusing the ``chain`` axis rather than
    adding a ``member`` one is what lets every existing reader keep working: a single-network
    run is just the one-member case, and pooling the chains (what
    :func:`pipeline.load_npe_posterior` does) gives the ensemble's combined posterior, while
    selecting one chain gives that member's own.
    """
    return np.concatenate(
        [stack_posterior_dict(samples, param_names) for samples in samples_by_member.values()],
        axis=0,
    )


def load_hdf5(filename):
    """Load a dictionary from a legacy HDF5 file (read-only compatibility shim)."""
    data_dict = {}
    with h5py.File(filename, "r") as file:
        for key, val in file.items():
            data_dict[key] = val[()] if val.shape is not None else None
    return data_dict
