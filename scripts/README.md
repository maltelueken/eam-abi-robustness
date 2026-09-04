# Pipeline scripts

One config-driven script per pipeline stage, shared by all four studies. Which test cases
a stage loops over comes from the `test_case` config group (`conf/test_case/`), so the
studies differ only in configuration, not in code.

| script | stage |
|---|---|
| `train_npe.py` | Train the neural posterior estimator and log recovery diagnostics. |
| `generate_test_data.py` | Simulate the held-out test datasets for every case. |
| `prior_stats.py` | Dump prior draws, for the prior ranges shown in the figures. |
| `predict_npe.py` | Sample the trained NPE's posterior for every case (one chain per ensemble member). |
| `fit_mcmc_gpu.py` | Fit ground-truth MCMC posteriors, vmapped over chains and datasets. |
| `check_metrics.py` | RMSE / contraction / calibration against the simulating parameters. |
| `check_summary_stats.py` | Posterior median and 95% credible interval per dataset. |
| `check_robustness.py` | Maximum mean discrepancy between the NPE and MCMC posteriors, per ensemble member. |
| `check_posterior_predictive.py` | Posterior predictive RT quantiles and accuracy. |
| `convert_hdf5_to_netcdf.py` | One-off migration for artifacts produced before the NetCDF switch. |

## Running

```console
python scripts/train_npe.py       experiment=experiment_1 model=rdm_simple
python scripts/fit_mcmc_gpu.py    experiment=experiment_2 model=rdm_simple_meta
python scripts/check_robustness.py experiment=experiment_4 model=rdm_simple
python scripts/generate_test_data.py experiment=experiment_3 model=rdm_sat
```

Every stage defaults to an **ensemble** of NPEs (`ensemble_size=5`); nothing has to be passed for
that. `approximator.ensemble_size=<n>` changes the member count, and
`approximator=continuous_approximator` falls back to a single network — which only the
architecture sweep does. See the "NPE ensembles" section of `CLAUDE.md`.

The sweep is the one stage run with `--multirun`, and on the cluster with `slurm/sweep.sh`
rather than `slurm/submit.sh`:

```console
python scripts/train_npe.py --multirun sweeper=optuna approximator=continuous_approximator \
    experiment=experiment_1 model=rdm_simple
```

Any `conf/**/*.yaml` key can be overridden on the command line. `fit_mcmc_gpu.py` accepts
`case=<key>` to fit a single test case (e.g. `case=sample_size_50`); by default it fits all
of them in one job.

Note that `hydra.run.dir` includes `hydra.job.override_dirname`, so **a stage must be given
the same overrides as the stage that produced its inputs** — otherwise it will look for the
trained model or the NPE samples in a different directory.

On the cluster, use `slurm/submit.sh` rather than calling these directly (and `slurm/sweep.sh`
for the architecture sweep).
