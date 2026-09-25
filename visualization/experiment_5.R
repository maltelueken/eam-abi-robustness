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

# Study 5 holds the prior fixed at study 2's full meta prior and manipulates the *data*
# instead: each training condition keeps only simulated datasets whose empirical accuracy
# falls in a band, and the test cases bin the same axis. So unlike studies 2 and 3 the
# question is not prior misspecification but amortization coverage -- how an approximator
# behaves on data regions it was never shown, at an unchanged prior.

study_labels <- c(
  TeX("A: 70--80\\% (medium)"),
  TeX("B: 50--60\\% (low)"),
  TeX("C: 90--100\\% (high)"),
  TeX("D: 50--100\\% (full)")
)

models <- c(
  "rdm_simple_meta_acc_medium",
  "rdm_simple_meta_acc_low",
  "rdm_simple_meta_acc_high",
  "rdm_simple_meta_acc_full"
)

# The accuracy band each model was trained on, from `simulator.acc_lower`/`acc_upper` in
# conf/model/rdm_simple_meta_acc_*.yaml.
df_training_band <- data.frame(
  study = models,
  x = c(0.70, 0.50, 0.90, 0.50),
  xend = c(0.80, 0.60, 1.00, 1.00)
) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

data_path <- "outputs/experiment_5"

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

ggsave(file.path(figure_path, "study_5_recovery.png"), width = 10, height = 7)


# Posterior mismatch across the accuracy bins ------------------------------

# `accuracy` is the realized accuracy of each dataset; `accuracy_lower`/`accuracy_upper` are
# the bin it was drawn from. Plotting against the realized value keeps the x-axis continuous
# while the shaded rectangle marks the band the model was trained on.
df_summary |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    median_diff = abs(mcmc_median - npe_median)
  ) |>
  ggplot(aes(x = accuracy, y = median_diff, color = study)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_rect(
    data = df_training_band,
    aes(xmin = x, xmax = xend, ymin = 0, ymax = Inf),
    inherit.aes = FALSE,
    fill = "grey",
    alpha = 0.3
  ) +
  geom_point(alpha = 0.15, size = 0.5) +
  geom_smooth(method = "loess", formula = y ~ x, se = FALSE, linewidth = 0.6) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(limits = c(0.5, 1.0), breaks = c(0.5, 0.75, 1.0)) +
  labs(x = "Accuracy of the generated data", y = "Absolute difference posterior median") +
  theme_half_open() +
  theme(
    legend.position = "none",
    strip.text.y = element_text(angle = 360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_5_posterior_mismatch_accuracy.png"), width = 10, height = 7)


# Distributional mismatch (MMD) --------------------------------------------

df_mmd |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    bin = sprintf("%.0f-%.0f", 100 * accuracy_lower, 100 * accuracy_upper)
  ) |>
  ggplot(aes(x = bin, y = mmd, color = study)) +
  facet_wrap(vars(study), nrow = 2) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = "Accuracy bin of the generated data (in %)",
    y = "Maximum mean discrepancy (NPE vs MCMC)"
  ) +
  theme_half_open() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 8, angle = -45, hjust = 0)
  )

ggsave(file.path(figure_path, "study_5_robustness_mmd.png"), width = 10, height = 5)
