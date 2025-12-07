

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

df_summary <- Reduce(rbind, lapply(models, function(mod) {
  read.csv(file.path(data_path, mod, "flow_matching/metrics/summary_stats.csv"))[,-1] |>
    mutate(study = mod)
}))

df_prior <- Reduce(rbind, lapply(models, function(mod) {
  read.csv(file.path(data_path, mod, "flow_matching/metrics/prior_stats.csv"))[,-1] |>
    mutate(study = mod)
}))

df_segment_prior <- df_prior |>
  pivot_longer(cols = c(v_intercept, v_slope, s_true, b, t0), names_to = "param") |>
  group_by(study, param) |>
  summarize(
    x = quantile(value, 0.001),
    xend = quantile(value, 0.999)
  ) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

df_range <- data.frame(
  param = rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2),
  true = c(0, 6, 0, 3, 0, 1.07, 0, 3, 0, 4),
  value = c(0, 6, 0, 3, 0, 1.07, 0, 3, 0, 4),
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

df_segment_prior <- data.frame(
  study = models,
  x = c(2.5, 0.5, 8.5, 0.5, 6.5, 0.5),
  xend = c(3.5, 1.5, 9.5, 3.5, 9.5, 9.5),
  y = c(2.5, 0.5, 8.5, 0.5, 6.5, 0.5),
  yend = c(3.5, 1.5, 9.5, 3.5, 9.5, 9.5)
) |>
  mutate(
    study = factor(study, levels = models, labels = study_labels)
  ) |>
  group_by(study, x, xend, y, yend) |>
  expand(param = c("v_slope", "b", "t0", "v_intercept", "s_true"))

df_summary |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    median_diff = abs(mcmc_median - npe_median),
    drift_slope_loc = factor(drift_slope_loc, labels = scales::number(unique(drift_slope_loc), accuracy = 0.01)),
    threshold_scale = factor(threshold_scale, labels = scales::number(unique(threshold_scale), accuracy = 0.001))
  ) |>
  group_by(study, param, threshold_scale, drift_slope_loc) |>
  summarise(median_diff = mean(median_diff)) |>
  ggplot(aes(x = threshold_scale, y = drift_slope_loc, fill = median_diff)) +
  facet_grid2(rows = vars(study), cols = vars(param)) +
  geom_raster() +
  geom_rect(
    data = df_segment_prior,
    mapping = aes(xmin = x, xmax = xend, ymin = y, ymax = yend),
    fill = NA,
    color = "white",
    alpha = 0.1,
    inherit.aes = FALSE
  ) +
  scale_fill_viridis_c() +
  labs(
    x = "Threshold scale",
    y = "Drift slope location",
    fill = "Absolute difference posterior median"
  ) +
  theme(
    axis.text = element_text(size = 8),
    axis.text.x = element_text(size = 8, angle = -45, hjust = 0),
    panel.background = element_blank(),
    legend.position = "top",
    legend.justification = "center",
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
