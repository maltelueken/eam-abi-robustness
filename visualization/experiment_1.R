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

study_labels <- c(
  TeX("A: 250, 500, 750, 1000"),
  TeX("B: 50, 100, ..., 250"),
  TeX("C: 1000, 1050, ..., 1200"),
  TeX("D: 50, 100, ..., 1200")
)

models <- c(
  "rdm_simple",
  "rdm_simple_discrete_lower",
  "rdm_simple_discrete_upper",
  "rdm_simple_discrete_full"
)

data_path <- "outputs/experiment_1"

df_summary <- read_by_model(data_path, models, "flow_matching/metrics/summary_stats.csv", read_summary_stats)

df_range <- data.frame(
  param = rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2),
  true = c(0, 3, 0, 3, 0, 1.07, 0, 4, 0, 4),
  value = c(0, 3, 0, 3, 0, 1.07, 0, 4, 0, 4),
  method = "MCMC"
)

df_summary |>
  pivot_longer(c(mcmc_median, npe_median), names_to = "method") |>
  mutate(
    study = factor(study, labels = study_labels),
    method = case_match(method, "mcmc_median" ~ "MCMC", .default = "NPE")
  ) |>
  ggplot(aes(x = true, y = value, color = method)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_point(alpha = 0.5, size = 0.5) +
  geom_abline(slope = 1, intercept = 0) +
  geom_blank(data = df_range) +
  labs(x = "True parameter", y = "Posterior median", color = "") +
  # scale_color_viridis_d() +
  scale_color_manual(values = c("indianred", "lightblue")) +
  theme_half_open() +
  theme(
    legend.position = "top",
    legend.justification = "center",
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_1_recovery.png"), width = 10, height = 6)

df_range <- data.frame(
  param = rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2),
  mcmc_median = c(0, 3, 0.5, 2.5, 0, 1.07, 0, 3, 0, 3),
  npe_median = c(0, 3, 0.5, 2.5, 0, 1.07, 0, 3, 0, 3)
)

df_summary |>
  filter(sample_size == 50) |>
  mutate(
    study = factor(study, labels = study_labels)
  ) |>
  ggplot(aes(x = mcmc_median, y = npe_median, color = study)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  geom_blank(data = df_range, mapping = aes(x = mcmc_median, y = npe_median), inherit.aes = FALSE) +
  labs(x = "MCMC posterior median", y = "NPE posterior median", color = "") +
  # scale_color_viridis_d() +
  scale_color_brewer(palette = "Dark2", labels = study_labels) +
  theme_half_open() +
  theme(
    legend.position = "none",
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_1_posterior_mismatch_50.png"), width = 10, height = 6)

df_range <- data.frame(
  param = rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2),
  sample_size = rep(c(50, 1250), each = 5),
  median_diff = c(0, 0.08, 0., 0.06, 0, 0.03, 0., 0.2, 0, 0.15),
  study = "study_a"
)

df_summary |>
  mutate(median_diff = abs(mcmc_median - npe_median)) |>
  group_by(sample_size, param, study) |>
  summarise(median_diff = median(median_diff)) |>
  ggplot(aes(x = sample_size, y = median_diff, color = study)) +
  facet_wrap(vars(param), ncol = 1, scales = "free_y") +
  geom_line() +
  geom_point() +
  geom_blank(data = df_range) +
  labs(
    x = "Trial number (test datasets)",
    y = "Absolute difference posterior median",
    color = "Trial number\n(training datasets)"
  ) +
  scale_x_continuous(breaks = c(50, 250, 500, 750, 1000, 1200), limits = c(50, 1200)) +
  scale_color_brewer(palette = "Dark2", labels = study_labels) +
  theme_half_open()

ggsave(file.path(figure_path, "study_1_posterior_mismatch.png"), width = 10, height = 8)

df_summary |>
  filter(sample_size == 50) |>
  group_by(param) |>
  summarise(
    mcmc_rmsd = sqrt(mean((true - mcmc_median)^2)),
    npe_rmsd = sqrt(mean((true - npe_median)^2))
  )
