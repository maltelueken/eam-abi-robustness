

library(ggplot2)
library(ggh4x)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"

study_labels <- paste("Context:", c("None", "High error rate", "Low error rate", "High + low error rate"))

data_path <- "outputs/experiment_3"


# Hyperparameter recovery -------------------------------------------------

df_hyperparams_b <- read.csv(file.path(data_path, "rdm_simple_meta_lower/flow_matching/summary_mmd/summary_mmd_params.csv"))[,-1] # Skip first index column
df_hyperparams_c <- read.csv(file.path(data_path, "rdm_simple_meta_upper/flow_matching/summary_mmd/summary_mmd_params.csv"))[,-1]
df_hyperparams_d <- read.csv(file.path(data_path, "rdm_simple_meta/flow_matching/summary_mmd/summary_mmd_params.csv"))[,-1]

df_hyperparams_join <- rbind(
  df_hyperparams_b |> mutate(study = "study_b"),
  df_hyperparams_c |> mutate(study = "study_c"),
  df_hyperparams_d |> mutate(study = "study_d")
) |>
  mutate(id = row_number()) |>
  pivot_longer(!c(study, id), names_pattern = "([^_]*)_(.*)", names_to = c("type", "param")) |>
  pivot_wider(id_cols = c(id, study, param), names_from = type)

param_labels <- c("mu_v_slope", "beta_b")

df_segment <- data.frame(
  x = c(0.5, 0.05, 2.83, 0.35, 0.5, 0.05),
  y = c(0.5, 0.05, 2.83, 0.35, 0.5, 0.05),
  x_end = c(1.67, 0.2, 4.0, 0.5, 4.0, 0.5),
  y_end = c(1.67, 0.2, 4.0, 0.5, 4.0, 0.5),
  study = rep(c("study_b", "study_c", "study_d"), each = 2),
  param = rep(c("drift_slope_loc", "threshold_scale"), 3)
) |>
  mutate(
    study = factor(study, labels = study_labels[-1]),
    param = factor(param, labels = param_labels)
  )

df_hyperparams_join |>
  mutate(
    study = factor(study, labels = study_labels[-1]),
    param = factor(param, labels = param_labels)
  ) |>
  ggplot(aes(x = true, y = est)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_segment(data = df_segment, aes(x = x, y = x, xend = x_end, yend = y_end), color = "red", size = 2, inherit.aes = FALSE) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "True parameter", y = "Estimated parameter") +
  theme_half_open() +
  theme(
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_3_hyperparams.png"))


# Generalization ----------------------------------------------------------

# Take simple condition from experiment 2
df_robustness_a <- read.csv(file.path("outputs/experiment_2", "rdm_simple/flow_matching/robustness/mmd.csv"))[,-1]

df_robustness_d <- read.csv(file.path(data_path, "rdm_simple_meta/flow_matching/robustness/mmd.csv"))[,-1]

df_robustness_b <- read.csv(file.path(data_path, "rdm_simple_meta_lower/flow_matching/robustness/mmd.csv"))[,-1]

df_robustness_c <- read.csv(file.path(data_path, "rdm_simple_meta_upper/flow_matching/robustness/mmd.csv"))[,-1]

df_summary_a <- read.csv(file.path("outputs/experiment_2", "rdm_simple/flow_matching/metrics/summary_stats.csv"))[,-1]

df_summary_d <- read.csv(file.path(data_path, "rdm_simple_meta/flow_matching/metrics/summary_stats.csv"))[,-1]

df_summary_b <- read.csv(file.path(data_path, "rdm_simple_meta_lower/flow_matching/metrics/summary_stats.csv"))[,-1]

df_summary_c <- read.csv(file.path(data_path, "rdm_simple_meta_upper/flow_matching/metrics/summary_stats.csv"))[,-1]

join_robustness_summary_dfs <- function(df_summary, df_robustness) {
  return(df_summary |>
           group_by(drift_slope_loc, threshold_scale, param) |>
           mutate(id = row_number(), .before = 1) |>
           left_join(df_robustness |>
                       group_by(drift_slope_loc, threshold_scale) |>
                       mutate(id = row_number()),
                     by=c("drift_slope_loc", "threshold_scale", "id")))
}

df_join_a <- join_robustness_summary_dfs(df_summary_a, df_robustness_a)
df_join_b <- join_robustness_summary_dfs(df_summary_b, df_robustness_b)
df_join_c <- join_robustness_summary_dfs(df_summary_c, df_robustness_c)
df_join_d <- join_robustness_summary_dfs(df_summary_d, df_robustness_d)

df_join <- rbind(
  df_join_a |> mutate(study = "study_a"),
  df_join_b |> mutate(study = "study_b"),
  df_join_c |> mutate(study = "study_c"),
  df_join_d |> mutate(study = "study_d")
)

df_range <- data.frame(
  param = rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2),
  true = c(0, 10, 0, 3, 0, 1.3, 0, 4, 0, 6),
  value = c(0, 10, 0, 3, 0, 1.3, 0, 4, 0, 6),
  method = "MCMC"
)

df_join |>
  pivot_longer(c(mcmc_median, npe_median), names_to = "method") |>
  mutate(
    study = factor(study, labels = study_labels),
    method = case_match(method, "mcmc_median" ~ "MCMC", .default = "NPE")
  ) |>
  ggplot(aes(x = value, y = true, color = method)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  geom_blank(data = df_range) +
  labs(x = "True parameter", y = "Posterior median", color = "") +
  theme_half_open() +
  theme(
    legend.position = "top",
    legend.justification = "center",
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_3_recovery.png"))

df_range <- data.frame(
  param = rep(rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2), 4),
  error_rate = 0,
  median_diff = rep(c(0, 3, 0, 1.25, 0, 0.8, 0, 1.5, 0, 1.5), 4),
  study = factor(rep(study_labels, each=10))
)

df_join |>
  mutate(
    study = factor(study, labels = study_labels),
    median_diff = abs(mcmc_median - npe_median)
    # study = factor(study, levels = rev(study_labels))
    # study = fct_rev(factor(study, levels = paste0("study_", c("a", "b", "c", "d")), labels = study_labels))
  ) |>
  ggplot(aes(x = 1-error_rate, y = median_diff, color = study)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_point(alpha = 0.1) +
  geom_smooth(xseq = seq(0, 0.5, 0.02), color = "black") +
  geom_blank(data = df_range) +
  labs(x = "Error rate", y = "Absolute difference posterior median", color = "Context-aware") +
  scale_x_continuous(limits = c(0, 0.5), breaks = c(0, 0.25, 0.5)) +
  theme_half_open() +
  theme(
    legend.position = "none",
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_3_posterior_mismatch.png"))
