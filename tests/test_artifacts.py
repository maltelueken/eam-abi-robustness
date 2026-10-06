"""Tests that every stage addresses an artifact by the same path."""

from artifacts import Artifacts
from cases import num_obs_grid, subject_files


def make_artifacts():
    return Artifacts({"test_data_path": "/shared/experiment_1/rdm_simple"}, run_root="/run/flow_matching")


def test_simulated_case_paths_split_between_the_shared_and_run_roots():
    artifacts = make_artifacts()
    case = num_obs_grid([50])[0]

    # Test data and MCMC do not depend on the approximator, so they are shared.
    assert artifacts.test_data(case) == "/shared/experiment_1/rdm_simple/test_data/test_data_sample_size_50.nc"
    assert artifacts.mcmc_samples(case) == "/shared/experiment_1/rdm_simple/mcmc_samples/fit_mcmc_sample_size_50.nc"

    # NPE samples and CSVs do.
    assert artifacts.npe_samples(case) == "/run/flow_matching/npe_samples/posterior_samples_sample_size_50.nc"
    assert artifacts.csv("metrics", "summary_stats") == "/run/flow_matching/metrics/summary_stats.csv"


def test_empirical_case_reads_its_source_file_in_place(tmp_path):
    (tmp_path / "FF1.txt").write_text("")
    case = subject_files(str(tmp_path))[0]

    assert make_artifacts().test_data(case) == str(tmp_path / "FF1.txt")
    # ...but its MCMC fit still lands in the shared root.
    assert make_artifacts().mcmc_samples(case).endswith("mcmc_samples/fit_mcmc_FF1.nc")


def test_ensure_creates_the_parent_directory(tmp_path):
    artifacts = Artifacts({"test_data_path": str(tmp_path)}, run_root=str(tmp_path))
    case = num_obs_grid([50])[0]

    artifacts.ensure(artifacts.mcmc_samples(case))

    assert (tmp_path / "mcmc_samples").is_dir()
