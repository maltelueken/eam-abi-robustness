"""Artifact paths for one pipeline run.

Two roots are in play, and conflating them was the source of a long-standing bug where
three scripts in the same directory each addressed the same MCMC file differently
(cwd-relative, `..`-relative, and under `test_data_path`):

* **shared root** -- `cfg["test_data_path"]`, i.e. `outputs/<experiment>/<model>`. Test data
  and MCMC fits live here because they do not depend on the inference method, so every
  approximator variant compares against the same held-out data and the same ground truth.
* **run root** -- the Hydra run directory (`hydra.job.chdir: True` makes it the cwd), i.e.
  `outputs/<experiment>/<model>/<inference-method>/...`. NPE samples and result CSVs live
  here because they *do* depend on the approximator.

Everything is addressed by `Case.key`, so writers and readers cannot drift apart.
"""

import os

TEST_DATA_DIR = "test_data"
MCMC_DIR = "mcmc_samples"
NPE_DIR = "npe_samples"

SUFFIX = ".nc"


class Artifacts:
    """Resolves artifact paths for one experiment/model/inference-method combination."""

    def __init__(self, cfg, run_root=None):
        self.shared_root = cfg["test_data_path"]
        self.run_root = os.getcwd() if run_root is None else run_root

    def test_data(self, case):
        """Path to a case's test data -- the empirical file itself when there is one."""
        if case.source_path is not None:
            return case.source_path

        return os.path.join(self.shared_root, TEST_DATA_DIR, f"test_data_{case.key}{SUFFIX}")

    def test_data_dir(self):
        """Directory holding the test data for this experiment/model."""
        return os.path.join(self.shared_root, TEST_DATA_DIR)

    def mcmc_samples(self, case):
        """Path to a case's MCMC posterior samples."""
        return os.path.join(self.shared_root, MCMC_DIR, f"fit_mcmc_{case.key}{SUFFIX}")

    def npe_samples(self, case):
        """Path to a case's NPE posterior samples."""
        return os.path.join(self.run_root, NPE_DIR, f"posterior_samples_{case.key}{SUFFIX}")

    def csv(self, subdir, name):
        """Path to a result CSV under this run's output directory."""
        return os.path.join(self.run_root, subdir, f"{name}.csv")

    def figure(self, name):
        """Path to a diagnostic figure under this run's output directory."""
        return os.path.join(self.run_root, name)

    def ensure(self, *paths):
        """Create the parent directory of each given path."""
        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
