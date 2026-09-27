# Loaders for the result tables written by scripts/check_*.py.
#
# Those scripts emit one long format for every study:
#
#   <case labels...>, dataset, param, method, statistic, value
#
# which keeps the schema identical whether or not a study has ground truth. The plotting
# code below works in the wide shape (npe_median, mcmc_lower, true, ...), so the pivot
# happens here, once, instead of in each figure script.

library(dplyr)
library(tidyr)

# Columns that are not case labels, in the long result tables.
.value_columns <- c("dataset", "param", "method", "statistic", "value")

#' Read a long-format summary-stats table and widen it to one row per dataset and parameter.
#'
#' Produces the columns the figure scripts expect: npe_median / npe_lower / npe_upper,
#' mcmc_median / mcmc_lower / mcmc_upper, and (for the simulation studies) true.
read_summary_stats <- function(path) {
  df <- read.csv(path)

  df |>
    mutate(column = ifelse(method == "true", "true", paste(method, statistic, sep = "_"))) |>
    select(-method, -statistic) |>
    pivot_wider(names_from = column, values_from = value)
}

#' Read a long-format prior-draw table (columns: draw, param, value).
read_prior_stats <- function(path) {
  read.csv(path)
}

#' Read the per-dataset MMD table (columns: <case labels...>, member, dataset, accuracy,
#' rt_min, rt_max, mmd).
#'
#' `member` indexes the ensemble member the NPE posterior came from; a single-network run has
#' one member and one row per dataset, as before. Aggregate over `member` for a run's overall
#' mismatch, or read its spread at a fixed dataset as the inference variability.
read_mmd <- function(path) {
  read.csv(path)
}

#' Read a long-format recovery-metrics table and widen it to one row per parameter.
read_metrics <- function(path) {
  read.csv(path) |>
    mutate(column = paste(method, statistic, sep = "_")) |>
    select(-method, -statistic, -dataset) |>
    pivot_wider(names_from = column, values_from = value)
}

#' Read the posterior-predictive table written by scripts/check_posterior_predictive.py.
read_ppd <- function(path) {
  read.csv(path)
}

#' Read one table for each model, tagging rows with the model name as `study`.
read_by_model <- function(data_path, models, relative_path, reader) {
  Reduce(rbind, lapply(models, function(mod) {
    reader(file.path(data_path, mod, relative_path)) |>
      mutate(study = mod)
  }))
}
