
import numpy as np
import pytest

import scipy.stats as stats

from pymc_models import inv_gauss_logpdf, inv_gauss_logsf


def test_inv_gauss_logpdf():
    x = np.array([-1, 0, 0.01, 0.5, 1.0, 2.0, 10.0, 100.0, 500.0, np.inf, -np.inf])

    mu = np.array([0.01, 1.0, 2.0, 10.0, 100.0])
    lam = np.array([0.01, 1.0, 2.0, 10.0, 100.0])

    for m in mu:
        for l in lam:
            ref = stats.invgauss.logpdf(x, mu=m/l, scale=l)
            res = inv_gauss_logpdf(x, m, l).eval()

            assert pytest.approx(res[np.isfinite(res)] == ref[np.isfinite(ref)])


# @pytest.mark.xfail()
def test_inv_gauss_logsf():
    x = np.array([-1, 0, 0.01, 0.5, 1.0, 2.0, 10.0, 100.0, 500.0, np.inf, -np.inf])

    mu = np.array([0.01, 1.0, 2.0, 10.0, 100.0])
    lam = np.array([0.01, 1.0, 2.0, 10.0, 100.0])

    for m in mu:
        for l in lam:
            ref = stats.invgauss.logsf(x, mu=2/2, scale=2)
            res = inv_gauss_logsf(x, 2, 2).eval()

            assert np.all(res == ref)