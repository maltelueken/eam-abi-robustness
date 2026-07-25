# Experiment 1 Python Scripts

- check_metrics.py: Computes and saves various metrics comparing posterior samples from NPE and MCMC methods.
- check_robustness.py: Assesses the robustness of inference by comparing posterior maximum mean discrepancy between MCMC and NPE across different test datasets.
- check_summary_stats.py: Calculates and saves summary statistics for posterior samples from NPE and MCMC.
- fit_mcmc_gpu.py: Runs MCMC inference for every test dataset of a given sample size in one GPU job, parallelized with `jax.vmap` across chains and datasets (see `rdm_jax.fit_mcmc_gpu_batch`).
- generate_test_data.py: Generates synthetic test datasets using the configured simulator.
- predict_npe.py: Runs NPE inference on test datasets and saves posterior samples.
- train_npe.py: Trains the NPE model using simulated data and configured hyperparameters.
