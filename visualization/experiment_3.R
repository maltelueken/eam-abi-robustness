source("visualization/load_results.R")


library(ggplot2)
library(ggh4x)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"


# Generalization ----------------------------------------------------------

# Study 3 simulates a speed-vs-accuracy manipulation: every dataset holds a speed-instructed
# and an accuracy-instructed block of trials, differing only in response threshold. The test
# cases sweep `threshold_diff_scale`, the scale of the prior on that threshold difference, so
# each model is evaluated on data drawn from priors it was and was not trained on.

study_labels <- c(
  TeX("A: medium-fixed"),
  TeX("B: low-fixed"),
  TeX("C: high-fixed"),
  TeX("D: low-varying"),
  TeX("E: high-varying"),
  TeX("F: full-varying")
)

models <- c(
  "rdm_sat",
  "rdm_sat_lower",
  "rdm_sat_upper",
  "rdm_sat_meta_lower",
  "rdm_sat_meta_upper",
  "rdm_sat_meta"
)

# The part of the swept range each model was trained on, from
# conf/simulator/prior_simulator/rdm_sat*.yaml and
# conf/simulator/meta_simulator/random_prior_meta_continuous_sat*.yaml. The fixed-prior models
# were trained at a single value, the varying ones over a range.
df_training_prior <- data.frame(
  study = models,
  x = c(0.10, 0.02, 0.18, 0.02, 0.14, 0.02),
  xend = c(0.10, 0.02, 0.18, 0.06, 0.18, 0.18)
) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

data_path <- "outputs/experiment_3"

df_summary <- read_by_model(data_path, models, "flow_matching/metrics/summary_stats.csv", read_summary_stats)

df_prior <- read_by_model(data_path, models, "flow_matching/metrics/prior_stats.csv", read_prior_stats)

df_mmd <- read_by_model(data_path, models, "flow_matching/robustness/mmd.csv", read_mmd)


# Parameter recovery ------------------------------------------------------

# prior_stats.csv is already long (draw, param, value), so no pivot is needed here.
df_segment_prior <- df_prior |>
  group_by(study, param) |>
  summarize(
    x = quantile(value, 0.001),
    xend = quantile(value, 0.999)
  ) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

df_summary |>
  pivot_longer(c(mcmc_median, npe_median), names_to = "method") |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    method = case_match(method, "mcmc_median" ~ "MCMC", .default = "NPE")
  ) |>
  ggplot(aes(x = true, y = value, color = method)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_point(alpha = 0.5, size = 0.5) +
  geom_abline(slope = 1, intercept = 0) +
  geom_rect(
    data = df_segment_prior,
    aes(xmin = x, xmax = xend, ymin = 0, ymax = Inf),
    inherit.aes = FALSE,
    fill = "grey",
    alpha = 0.3
  ) +
  scale_color_manual(values = c("indianred", "lightblue")) +
  labs(x = "True parameter", y = "Posterior median", color = "") +
  theme_half_open() +
  theme(
    legend.position = "top",
    legend.justification = "center",
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_3_recovery.png"), width = 10, height = 7)


# Posterior mismatch across the swept prior --------------------------------

# The one-dimensional counterpart of study 2's heatmap: one hyperparameter is swept, so the
# mismatch can be shown directly against it, with the training prior marked underneath.
df_summary |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    median_diff = abs(mcmc_median - npe_median)
  ) |>
  group_by(study, param, threshold_diff_scale) |>
  summarise(median_diff = mean(median_diff)) |>
  ggplot(aes(x = threshold_diff_scale, y = median_diff, color = study)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_rect(
    data = df_training_prior,
    aes(xmin = x, xmax = xend, ymin = 0, ymax = Inf),
    inherit.aes = FALSE,
    fill = "grey",
    alpha = 0.3
  ) +
  geom_vline(
    data = filter(df_training_prior, x == xend),
    aes(xintercept = x),
    linetype = "dashed",
    color = "grey40"
  ) +
  geom_line() +
  geom_point(size = 1) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = "Scale of the prior on the threshold difference",
    y = "Absolute difference posterior median"
  ) +
  theme_half_open() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 8, angle = -45, hjust = 0),
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_3_posterior_mismatch_prior.png"), width = 10, height = 7)


# Distributional mismatch (MMD) --------------------------------------------

df_mmd |>
  mutate(study = factor(study, levels = models, labels = study_labels)) |>
  ggplot(aes(x = factor(threshold_diff_scale), y = mmd, color = study)) +
  facet_wrap(vars(study), nrow = 2) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = "Scale of the prior on the threshold difference",
    y = "Maximum mean discrepancy (NPE vs MCMC)"
  ) +
  theme_half_open() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 8, angle = -45, hjust = 0)
  )

ggsave(file.path(figure_path, "study_3_robustness_mmd.png"), width = 10, height = 5)
