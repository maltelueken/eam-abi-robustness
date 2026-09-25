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

# Study 6 holds the prior fixed at study 2's full meta prior and manipulates the *data*, as
# study 5 does, but along the response-time axis: each training condition keeps only simulated
# datasets whose response times all fall in a window, and the test cases reuse the same four
# windows. So the question is again amortization coverage -- how an approximator behaves on
# data regions it was never shown, at an unchanged prior -- not prior misspecification.
#
# Unlike study 5's accuracy bins the windows are nested, so a wide test case's draw contains
# datasets a narrow one would also have accepted. The continuous axis below is therefore each
# dataset's realized slowest response time (`rt_max`, written per dataset by
# scripts/check_robustness.py and check_summary_stats.py), not the window it was drawn from.

study_labels <- c(
  TeX("A: $rt \\in [0.15, 1]$ (tight)"),
  TeX("B: $rt \\in [0.15, 2]$ (medium)"),
  TeX("C: $rt \\in [0.15, 4]$ (wide)"),
  TeX("D: $rt \\in [0, 20]$ (full)")
)

models <- c(
  "rdm_simple_meta_rt_tight",
  "rdm_simple_meta_rt_medium",
  "rdm_simple_meta_rt_wide",
  "rdm_simple_meta_rt_full"
)

# The response-time window each model was trained on, from `simulator.rt_lower`/`rt_upper` in
# conf/model/rdm_simple_meta_rt_*.yaml. Since every response time had to fall inside it, the
# window is also the range of *slowest* response times the model ever saw.
df_training_window <- data.frame(
  study = models,
  x = c(0.15, 0.15, 0.15, 0.01),
  xend = c(1.0, 2.0, 4.0, 20.0)
) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

data_path <- "outputs/experiment_6"

df_summary <- read_by_model(data_path, models, "flow_matching/metrics/summary_stats.csv", read_summary_stats)

df_mmd <- read_by_model(data_path, models, "flow_matching/robustness/mmd.csv", read_mmd)


# Parameter recovery ------------------------------------------------------

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
  scale_color_manual(values = c("indianred", "lightblue")) +
  labs(x = "True parameter", y = "Posterior median", color = "") +
  theme_half_open() +
  theme(
    legend.position = "top",
    legend.justification = "center",
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_6_recovery.png"), width = 10, height = 7)


# Posterior mismatch across the response-time range -------------------------

# `rt_max` is the realized slowest response time of each dataset; `rt_lower`/`rt_upper` are the
# window it was drawn from. Plotting against the realized value keeps the x-axis continuous
# while the shaded rectangle marks the window the model was trained on -- everything to its
# right is data the approximator never saw. Log scale: the slowest response time spans two
# orders of magnitude under this prior.
df_summary |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    median_diff = abs(mcmc_median - npe_median)
  ) |>
  ggplot(aes(x = rt_max, y = median_diff, color = study)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_rect(
    data = df_training_window,
    aes(xmin = x, xmax = xend, ymin = 0, ymax = Inf),
    inherit.aes = FALSE,
    fill = "grey",
    alpha = 0.3
  ) +
  geom_point(alpha = 0.15, size = 0.5) +
  geom_smooth(method = "loess", formula = y ~ x, se = FALSE, linewidth = 0.6) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_log10(breaks = c(0.5, 1, 2, 4, 10, 20)) +
  labs(
    x = "Slowest response time of the generated data (s)",
    y = "Absolute difference posterior median"
  ) +
  theme_half_open() +
  theme(
    legend.position = "none",
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_6_posterior_mismatch_rt.png"), width = 10, height = 7)


# Distributional mismatch (MMD) --------------------------------------------

df_mmd |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    window = sprintf("[%.2f, %.0f]", rt_lower, rt_upper)
  ) |>
  ggplot(aes(x = window, y = mmd, color = study)) +
  facet_wrap(vars(study), nrow = 2) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = "Response-time window of the generated data (s)",
    y = "Maximum mean discrepancy (NPE vs MCMC)"
  ) +
  theme_half_open() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 8, angle = -45, hjust = 0)
  )

ggsave(file.path(figure_path, "study_6_robustness_mmd.png"), width = 10, height = 5)


# The same mismatch, re-binned by realized slowest response time -------------

# The nested test windows overlap, which mutes the boxplot above: most of what the [0, 20] case
# draws would also have passed [0.15, 2]. Re-binning the widest case's datasets by their own
# `rt_max` recovers disjoint bins, and with them the comparison the accuracy bins of study 5
# give directly.
df_mmd |>
  filter(rt_upper == 20) |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    bin = cut(rt_max, breaks = c(0, 1, 2, 4, 8, Inf), labels = c("<1", "1-2", "2-4", "4-8", ">8"))
  ) |>
  ggplot(aes(x = bin, y = mmd, color = study)) +
  facet_wrap(vars(study), nrow = 2) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = "Slowest response time of the generated data (s)",
    y = "Maximum mean discrepancy (NPE vs MCMC)"
  ) +
  theme_half_open() +
  theme(legend.position = "none")

ggsave(file.path(figure_path, "study_6_robustness_mmd_by_slowest_rt.png"), width = 10, height = 5)
