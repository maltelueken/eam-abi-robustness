# Experiment 2 Python Scripts

- check_metrics.py: Computes metrics for posterior samples across meta-parameter grids using NPE and MCMC.
- check_robustness.py: Evaluates the maximum mean discrepancy between MCMC and NPE over meta-parameter combinations.
- check_summary_stats.py: Calculates summary statistics for posterior samples across meta-parameter grids.
- collect_mcmc.py: Aggregates MCMC samples for different meta-parameter settings.
- fit_mcmc_slurm.py: Runs MCMC inference for specific meta-parameter combinations, suitable for SLURM jobs.
- generate_test_data.py: Generates test datasets for various meta-parameter combinations.
- predict_npe.py: Runs NPE inference and saves posterior samples for each meta-parameter setting.