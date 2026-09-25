"""Convert the pipeline's legacy HDF5 artifacts to NetCDF.

The published `outputs/` archives predate the NetCDF switch. MCMC fits in particular are
expensive to reproduce, so convert them in place rather than refitting.

    python scripts/convert_hdf5_to_netcdf.py outputs/experiment_1/rdm_simple \
        --mcmc-params v_intercept v_slope s_true b t0

Legacy MCMC files store `(dataset, chain, draw, param)`; NetCDF stores
`(chain, draw, dataset, param)`, so the parameter names must be supplied -- they are not
recorded in the HDF5 file.
"""

import argparse
import logging
import os
import sys
from glob import glob

import numpy as np

from data import load_hdf5, save_dataset, save_posterior

logger = logging.getLogger(__name__)


def convert_dataset(path, target):
    """Convert one simulator-output file."""
    save_dataset(target, load_hdf5(path))


def convert_posterior_file(path, target, param_names, source_layout):
    """Convert one posterior file to the `(chain, draw, dataset, param)` NetCDF layout."""
    contents = load_hdf5(path)

    if "samples" in contents:
        # MCMC: a single (dataset, chain, draw, param) array.
        samples = np.transpose(contents["samples"], (1, 2, 0, 3))
    else:
        # NPE: one (dataset, draw, 1) entry per parameter.
        columns = [np.asarray(contents[name]).squeeze(axis=-1) for name in param_names]
        samples = np.transpose(np.stack(columns, axis=-1), (1, 0, 2))[np.newaxis, ...]

    if samples.shape[-1] != len(param_names):
        msg = f"{path}: stored {samples.shape[-1]} parameters but {len(param_names)} names were given."
        raise ValueError(msg)

    logger.info("%s %s -> %s", source_layout, path, target)
    save_posterior(target, samples, param_names)


def convert_tree(root, mcmc_params, npe_params, *, dry_run=False):
    """Convert every recognised HDF5 artifact beneath `root`."""
    converted = 0

    for path in sorted(glob(os.path.join(root, "**", "*.hdf5"), recursive=True)):
        target = os.path.splitext(path)[0] + ".nc"
        parent = os.path.basename(os.path.dirname(path))
        stem = os.path.basename(path)

        # Match on the artifact prefix, not just the directory: `mcmc_samples/` also holds
        # per-dataset `samples_*.hdf5` leftovers from the removed SLURM-array fitting
        # scripts, which have a different shape and are not part of the pipeline any more.
        if parent == "test_data" and stem.startswith("test_data_"):
            kind, convert = "test data", convert_dataset
        elif parent == "mcmc_samples" and stem.startswith("fit_mcmc_"):
            kind, convert = "mcmc", None
        elif parent == "npe_samples" and stem.startswith("posterior_samples_"):
            kind, convert = "npe", None
        else:
            logger.debug("skipping %s", path)
            continue

        if dry_run:
            logger.info("would convert (%s) %s -> %s", kind, path, target)
            converted += 1
            continue

        if convert is not None:
            convert(path, target)
        else:
            convert_posterior_file(path, target, mcmc_params if kind == "mcmc" else npe_params, kind)

        converted += 1

    return converted


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="directory to search for .hdf5 files")
    parser.add_argument("--mcmc-params", nargs="+", required=True, help="parameter names stored in MCMC files")
    parser.add_argument("--npe-params", nargs="+", help="parameter names stored in NPE files (default: --mcmc-params)")
    parser.add_argument("--dry-run", action="store_true", help="list what would be converted and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    converted = convert_tree(
        args.root,
        args.mcmc_params,
        args.npe_params or args.mcmc_params,
        dry_run=args.dry_run,
    )

    logger.info("Converted %s file(s).", converted)


if __name__ == "__main__":
    sys.exit(main())
