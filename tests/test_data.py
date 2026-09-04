"""Round-trip tests for the NetCDF artifact layer, plus the legacy HDF5 reader."""

import h5py
import numpy as np
import pytest
from data import (
    load_dataset,
    load_hdf5,
    load_posterior,
    save_dataset,
    save_posterior,
    stack_posterior_dict,
)

PARAM_NAMES = ["v_intercept", "v_slope", "s_true", "b", "t0"]


def make_simulator_output(num_datasets=6, num_obs=20):
    """The shape a CustomSimulator actually returns: (batch, 1) draws, (batch, obs, 2) data."""
    rng = np.random.default_rng(0)
    output = {name: rng.random((num_datasets, 1)) for name in PARAM_NAMES}
    output["x"] = rng.random((num_datasets, num_obs, 2))
    output["num_obs"] = np.array(num_obs)
    return output


def test_dataset_round_trip_preserves_shapes_and_values(tmp_path):
    original = make_simulator_output()
    path = str(tmp_path / "test_data.nc")

    save_dataset(path, original)
    loaded = load_dataset(path)

    assert set(loaded) == set(original)
    for key, value in original.items():
        # Trailing singleton axes must survive: BayesFlow's adapter is shape-sensitive.
        assert loaded[key].shape == value.shape, key
        assert np.array_equal(loaded[key], value), key


def test_dataset_round_trip_keeps_scalars_zero_dimensional(tmp_path):
    path = str(tmp_path / "scalars.nc")
    save_dataset(path, {"num_obs": np.array(50), "drift_slope_loc": np.array(1.28)})

    loaded = load_dataset(path)

    assert loaded["num_obs"].shape == ()
    assert loaded["drift_slope_loc"] == pytest.approx(1.28)


def test_dataset_rejects_an_array_whose_rank_contradicts_its_declared_dims(tmp_path):
    with pytest.raises(ValueError, match="rank"):
        save_dataset(str(tmp_path / "bad.nc"), {"x": np.zeros((2, 3))})


def test_posterior_round_trip_keeps_dims_and_param_coords(tmp_path):
    path = str(tmp_path / "posterior.nc")
    samples = np.random.default_rng(1).random((4, 50, 6, len(PARAM_NAMES)))

    save_posterior(path, samples, PARAM_NAMES)
    loaded = load_posterior(path)

    assert loaded["theta"].dims == ("chain", "draw", "dataset", "param")
    assert list(loaded.coords["param"].to_numpy()) == PARAM_NAMES
    assert np.allclose(loaded["theta"].to_numpy(), samples)
    # Selecting by name is the point of storing coords at all.
    assert np.allclose(loaded["theta"].sel(param="t0").to_numpy(), samples[..., -1])


def test_posterior_rejects_a_name_list_of_the_wrong_length(tmp_path):
    with pytest.raises(ValueError, match="parameters"):
        save_posterior(str(tmp_path / "bad.nc"), np.zeros((1, 2, 3, 5)), ["a", "b"])


def test_stack_posterior_dict_matches_the_bayesflow_output_layout():
    num_datasets, num_draws = 6, 30
    samples = {name: np.random.default_rng(2).random((num_datasets, num_draws, 1)) for name in PARAM_NAMES}

    theta = stack_posterior_dict(samples, PARAM_NAMES)

    # NPE draws are independent, so they are stored as a single chain.
    assert theta.shape == (1, num_draws, num_datasets, len(PARAM_NAMES))
    assert np.allclose(theta[0, :, :, 0], samples["v_intercept"][:, :, 0].T)


def test_load_hdf5_still_reads_the_legacy_format(tmp_path):
    path = str(tmp_path / "legacy.hdf5")
    payload = {"samples": np.arange(24.0).reshape(2, 3, 4)}

    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples", data=payload["samples"])

    assert np.array_equal(load_hdf5(path)["samples"], payload["samples"])
