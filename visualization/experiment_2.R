source("visualization/load_results.R")



library(ggplot2)
library(ggh4x)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)
library(truncnorm)

figure_path <- "visualization/figures/"


# Generalization ----------------------------------------------------------

study_labels <- c(
  TeX("A: medium-fixed"),
  TeX("B: high-fixed"),
  TeX("C: low-fixed"),
  TeX("D: high-varying"),
  TeX("E: low-varying"),
  TeX("F: full-varying")
)

models <- c(
  "rdm_simple",
  "rdm_simple_lower",
  "rdm_simple_upper",
  "rdm_simple_meta_lower",
  "rdm_simple_meta_upper",
  "rdm_simple_meta"
)

data_path <- "outputs/experiment_2"

df_summary <- read_by_model(data_path, models, "flow_matching/metrics/summary_stats.csv", read_summary_stats)

df_prior <- read_by_model(data_path, models, "flow_matching/metrics/prior_stats.csv", read_prior_stats)

# prior_stats.csv is already long (draw, param, value), so no pivot is needed here.
df_segment_prior <- df_prior |>
  group_by(study, param) |>
  summarize(
    x = quantile(value, 0.001),
    xend = quantile(value, 0.999)
  ) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

# `v_slope` runs to 6 rather than 4: the sweep now reaches drift_slope_loc = 4.0, so with a
# prior scale of 0.5 the drawn slopes reach about 5.5.
df_range <- data.frame(
  param = rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2),
  true = c(0, 6, 0, 3, 0, 1.07, 0, 3, 0, 6),
  value = c(0, 6, 0, 3, 0, 1.07, 0, 3, 0, 6),
  method = "MCMC"
)

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
  geom_blank(data = df_range) +
  # scale_color_viridis_d() +
  scale_color_manual(values = c("indianred", "lightblue")) +
  labs(x = "True parameter", y = "Posterior median", color = "") +
  theme_half_open() +
  theme(
    legend.position = "top",
    legend.justification = "center",
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_2_recovery.png"), width = 10, height = 7)

# The prior each model was trained on, in `drift_slope_loc` units. The three "-fixed" models pin
# a single value (conf/simulator/prior_simulator/rdm_{simple,lower,upper}.yaml); the three
# "-varying" ones are amortized over a range (the bounds in conf/simulator/meta_simulator/
# random_prior_meta_continuous_multivariate*.yaml). Data coordinates, not factor positions --
# study 2 sweeps one continuous hyperparameter now, so the x axis is the hyperparameter itself.
df_trained_range <- data.frame(
  study = models,
  xmin = c(1.5, 0.7, 3.9, 0.7, 3.5, 0.7),
  xmax = c(1.5, 0.7, 3.9, 1.1, 3.9, 3.9)
) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

df_trained_band <- filter(df_trained_range, xmin < xmax)
df_trained_point <- filter(df_trained_range, xmin == xmax)

df_summary |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    median_diff = abs(mcmc_median - npe_median)
  ) |>
  group_by(study, param, drift_slope_loc) |>
  summarise(median_diff = mean(median_diff), .groups = "drop") |>
  ggplot(aes(x = drift_slope_loc, y = median_diff)) +
  # Rows are parameters so that `scales = "free_y"` gives each parameter its own range while
  # keeping the studies (columns) directly comparable within it.
  facet_grid2(rows = vars(param), cols = vars(study), scales = "free_y") +
  geom_rect(
    data = df_trained_band,
    mapping = aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
    fill = "grey",
    alpha = 0.3,
    inherit.aes = FALSE
  ) +
  geom_vline(
    data = df_trained_point,
    mapping = aes(xintercept = xmin),
    color = "grey40",
    linetype = "dashed"
  ) +
  geom_line() +
  geom_point(size = 1) +
  labs(
    x = "Drift slope prior location",
    y = "Absolute difference posterior median"
  ) +
  theme_half_open() +
  theme(
    axis.text = element_text(size = 8),
    panel.background = element_blank(),
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_2_posterior_mismatch_prior.png"), width = 10, height = 7)

df_range <- data.frame(
  param = rep(rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2), 6),
  acc = 0,
  median_diff = rep(c(0, 3, 0, 1.25, 0, 0.8, 0, 1.5, 0, 1.5), 6),
  study = factor(rep(models, each=10), levels = models, labels = study_labels)
)

df_segment_error_rate <- df_prior |>
  group_by(study) |>
  summarize(
    x = 100 * quantile(acc, 0.001),
    xend = min(100 * quantile(acc, 0.999), 50)
  ) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

df_summary |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    median_diff = abs(mcmc_median - npe_median)
  ) |>
  ggplot(aes(x = (1-acc)*100, y = median_diff, color = study)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_point(alpha = 0.1) +
  geom_rect(
    data = df_segment_error_rate,
    aes(xmin = x, xmax = xend, ymin = 0, ymax = Inf),
    inherit.aes = FALSE,
    fill = "grey",
    alpha = 0.3
  ) +
  geom_blank(data = df_range) +
  labs(x = "Error rate (in %)", y = "Absolute difference posterior median", color = "Context-aware") +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(limits = c(0, 50), breaks = c(0, 25, 50)) +
  theme_half_open() +
  theme(
    legend.position = "none",
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_2_posterior_mismatch_error_rate.png"), width = 10, height = 7)
