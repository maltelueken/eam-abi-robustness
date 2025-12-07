# Experiment 1 Python Scripts

- check_metrics.py: Computes and saves various metrics comparing posterior samples from NPE and MCMC methods.
- check_robustness.py: Assesses the robustness of inference by comparing posterior maximum mean discrepancy between MCMC and NPE across different test datasets.
- check_summary_stats.py: Calculates and saves summary statistics for posterior samples from NPE and MCMC.
- collect_mcmc.py: Collects and aggregates MCMC samples from multiple test datasets for analysis.
- fit_mcmc_slurm.py: Runs MCMC inference for a specific test dataset, designed for SLURM job arrays.
- generate_test_data.py: Generates synthetic test datasets using the configured simulator.
- predict_npe.py: Runs NPE inference on test datasets and saves posterior samples.
- train_npe.py: Trains the NPE model using simulated data and configured hyperparameters.
