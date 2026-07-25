# Experiment 4 Python Scripts

- check_metrics.py: Computes and saves metrics comparing NPE and MCMC posterior samples for each test file.
- check_posterior_predictive.py: Calculates posterior predictive checks using NPE and simulator for each test file.
- check_robustness.py: Assesses the maximum mean discrepancy between MCMC and NPE for each test file using NPE and MCMC samples.
- check_summary_stats.py: Computes summary statistics for posterior samples for each test file.
- fit_mcmc_gpu.py: Runs MCMC inference for every subject in a test file in one GPU job, parallelized with `jax.vmap` across chains and subjects (see `rdm_jax.fit_mcmc_gpu_batch`).
- predict_npe.py: Runs NPE inference and saves posterior samples for each test file.