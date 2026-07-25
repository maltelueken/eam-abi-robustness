# Experiment 2 Python Scripts

- check_metrics.py: Computes metrics for posterior samples across meta-parameter grids using NPE and MCMC.
- check_robustness.py: Evaluates the maximum mean discrepancy between MCMC and NPE over meta-parameter combinations.
- check_summary_stats.py: Calculates summary statistics for posterior samples across meta-parameter grids.
- fit_mcmc_gpu.py: Runs MCMC inference for every test dataset of a given meta-parameter combination in one GPU job, parallelized with `jax.vmap` across chains and datasets (see `rdm_jax.fit_mcmc_gpu_batch`).
- generate_test_data.py: Generates test datasets for various meta-parameter combinations.
- predict_npe.py: Runs NPE inference and saves posterior samples for each meta-parameter setting.