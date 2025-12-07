# Experiment 4 Python Scripts

- check_metrics.py: Computes and saves metrics comparing NPE and MCMC posterior samples for each test file.
- check_posterior_predictive.py: Calculates posterior predictive checks using NPE and simulator for each test file.
- check_robustness.py: Assesses the maximum mean discrepancy between MCMC and NPE for each test file using NPE and MCMC samples.
- check_summary_stats.py: Computes summary statistics for posterior samples for each test file.
- collect_mcmc.py: Collects and aggregates MCMC samples for each test file.
- fit_mcmc_slurm.py: Runs MCMC inference for a specific test file, designed for SLURM job arrays.
- predict_npe.py: Runs NPE inference and saves posterior samples for each test file.