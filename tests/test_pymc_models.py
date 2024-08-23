
import numpy as np
import pytest

import scipy.stats as stats

from pymc_models import inv_gauss_logpdf, inv_gauss_logsf, standard_normal_logcdf


@pytest.fixture()
def x():
    return np.array([-1, 0, 0.01, 0.5, 1.0, 2.0, 10.0, 100.0, 500.0, np.inf, -np.inf])


def test_inv_gauss_logpdf(x):
    mu = np.array([0.01, 1.0, 2.0, 10.0, 100.0])
    lam = np.array([0.01, 1.0, 2.0, 10.0, 100.0])

    for m in mu:
        for l in lam:
            ref = stats.invgauss.logpdf(x, mu=m/l, scale=l)
            res = inv_gauss_logpdf(x, m, l).eval()

            assert np.all(pytest.approx(res[np.isfinite(res)]) == ref[np.isfinite(ref)])


def test_standard_normal_logcdf(x):
    ref = stats.norm.logcdf(x)
    res = standard_normal_logcdf(x).eval()
    assert np.all(pytest.approx(res) == ref)


# @pytest.mark.xfail()
def test_inv_gauss_logsf(x):

    mu = np.array([0.01, 1.0, 2.0, 10.0, 100.0])
    lam = np.array([0.01, 1.0, 2.0, 10.0, 100.0])

    for m in mu:
        for l in lam:
            ref = stats.invgauss.logsf(x, mu=m/l, scale=l)
            res = inv_gauss_logsf(x, m, l).eval()

            assert np.all(pytest.approx(ref) == res)
