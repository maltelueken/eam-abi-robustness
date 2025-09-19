

library(ggplot2)
library(ggh4x)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"


# Generalization ----------------------------------------------------------

study_labels <- paste("Context:", c("None", "High error rate", "Low error rate", "High + low error rate"))

data_path <- "outputs/experiment_2"

df_robustness_a <- read.csv(file.path(data_path, "rdm_simple/flow_matching/robustness/mmd.csv"))[,-1] # Skip first index column

df_robustness_d <- read.csv(file.path(data_path, "rdm_simple_meta/flow_matching/robustness/mmd.csv"))[,-1]

df_robustness_b <- read.csv(file.path(data_path, "rdm_simple_meta_lower/flow_matching/robustness/mmd.csv"))[,-1]

df_robustness_c <- read.csv(file.path(data_path, "rdm_simple_meta_upper/flow_matching/robustness/mmd.csv"))[,-1]

df_summary_a <- read.csv(file.path(data_path, "rdm_simple/flow_matching/metrics/summary_stats.csv"))[,-1]

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

ggsave(file.path(figure_path, "study_2_recovery.png"))

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

ggsave(file.path(figure_path, "study_2_posterior_mismatch.png"))

df_median_mmd <- df_robustness |>
  group_by(study, drift_slope_loc, threshold_scale) |>
  summarise(mmd = median(sqrt(mmd))) |>
  mutate(
    drift_slope_loc = factor(drift_slope_loc),
    threshold_scale = factor(threshold_scale)
  )

x_label <- TeX("Prior scale ($\\beta_b$)")
y_label <- TeX("Prior location ($\\mu_{v_{slope}}$)")
fill_label <- "Posterior mismatch (median MMD)"

theme_raster <- theme(
  text = element_text(size = 14),
  axis.text = element_text(size = 10),
  panel.background = element_blank(),
  legend.direction = "horizontal",
  legend.justification = "center"
)

p1 <- df_median_mmd |>
  filter(study == "study_a") |>
  ggplot(aes(x = threshold_scale, y = drift_slope_loc, fill = mmd)) +
  facet_wrap(vars("A: Context-aware - No")) +
  geom_raster() +
  annotate("point", size = 3, x = 3, y = 3.75, shape = 8, color = "white") +
  labs(x = x_label, y = y_label, fill = fill_label) +
  scale_fill_viridis_c(limits =c(0.1, 0.65)) +
  theme_raster

p2 <- df_median_mmd |>
  filter(study == "study_d") |>
  ggplot(aes(x = threshold_scale, y = drift_slope_loc, fill = mmd)) +
  facet_wrap(vars("D: Context-aware - Full")) +
  geom_raster() +
  labs(x = x_label, y = "", fill = fill_label) +
  scale_fill_viridis_c(limits =c(0.1, 0.65)) +
  theme_raster

p3 <- df_robustness |>
  group_by(study, drift_slope_loc, threshold_scale) |>
  summarise(error_rate = median(1-error_rate)) |>
  mutate(
    drift_slope_loc = factor(drift_slope_loc),
    threshold_scale = factor(threshold_scale)
  ) |>
  ggplot(aes(x = threshold_scale, y = drift_slope_loc, fill = error_rate)) +
  facet_wrap(vars("Error rate")) +
  geom_raster() +
  geom_segment(aes(x = 0.5, xend = 4.5, y = 4.5)) +
  geom_segment(aes(y = 0.5, yend = 4.5, x = 4.5)) +
  geom_segment(aes(x = 6.5, xend = 10.5, y = 6.5)) +
  geom_segment(aes(y = 6.5, yend = 10.5, x = 6.5)) +
  annotate("point", size = 3, x = 3, y = 3.75, shape = 8, color = "white") +
  labs(x = x_label, y = "", fill = "Median error rate") +
  scale_fill_viridis_c(limits = c(0, 0.4), begin = 0.1, option = "magma") +
  theme_raster

plot_grid(
  plot_grid(get_legend(p1), get_legend(p3), ncol = 2),
  plot_grid(
    p1 + theme(legend.position = "none"),
    p2 + theme(legend.position = "none"),
    p3 + theme(legend.position = "none"),
    ncol = 3
  ),
  ncol = 1,
  rel_heights = c(0.2, 1)
)

p4 <- df_median_mmd |>
  filter(study == "study_b") |>
  ggplot(aes(x = threshold_scale, y = drift_slope_loc, fill = mmd)) +
  facet_wrap(vars("B: Context-aware - High error rate")) +
  geom_raster() +
  # annotate("rect", xmin = 0.5, xmax = 4.5, ymin = 0.5, ymax = 4.5, alpha = 0.3, fill = "darkred") +
  geom_segment(aes(x = 0.5, xend = 4.5, y = 4.5)) +
  geom_segment(aes(y = 0.5, yend = 4.5, x = 4.5)) +
  labs(x = x_label, y = y_label, fill = fill_label) +
  scale_fill_viridis_c(limits =c(0.1, 1.3), breaks = seq(0.1, 1.3, 0.3)) +
  theme_raster

p5 <- df_median_mmd |>
  filter(study == "study_c") |>
  ggplot(aes(x = threshold_scale, y = drift_slope_loc, fill = mmd)) +
  facet_wrap(vars("C: Context-aware - Low error rate")) +
  geom_raster() +
  geom_segment(aes(x = 6.5, xend = 10.5, y = 6.5)) +
  geom_segment(aes(y = 6.5, yend = 10.5, x = 6.5)) +
  labs(x = x_label, y = "", fill = fill_label) +
  scale_fill_viridis_c(limits =c(0.1, 1.3)) +
  theme_raster

plot_grid(
  plot_grid(
    plot_grid(get_legend(p1), get_legend(p3), ncol = 2),
    plot_grid(
      p1 + theme(legend.position = "none"),
      p2 + theme(legend.position = "none"),
      p3 + theme(legend.position = "none"),
      ncol = 3
    ),
    ncol = 1,
    rel_heights = c(0.2, 1)
  ),
  plot_grid(
    get_legend(p4),
    plot_grid(
      p4 + theme(legend.position = "none"),
      p5 + theme(legend.position = "none"),
      ncol = 2
    ),
    ncol = 1,
    rel_heights = c(0.2, 1)
  ),
  ncol = 1
)

ggsave(file.path(figure_path, "study_2_posterior_mmd_prior.png"), width = 10, height = 8)

df_robustness |>
  ggplot(aes(x = 1-error_rate, y = sqrt(mmd), color = study)) +
  geom_point(alpha = 0.05) +
  geom_smooth(xseq = seq(0, 0.5, 0.02), ) +
  labs(x = "Error rate", y = "Posterior mismatch (MMD)", color = "Context-aware") +
  xlim(c(0, 0.5)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open()

ggsave(file.path(figure_path, "study_2_posterior_mmd_error.png"))


# Metrics -----------------------------------------------------------------

df_metrics_a <- read.csv("outputs/experiment_2/rdm_simple/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_b <- read.csv("outputs/experiment_2/rdm_simple_meta_lower/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_c <- read.csv("outputs/experiment_2/rdm_simple_meta_upper/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_d <- read.csv("outputs/experiment_2/rdm_simple_meta/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics <- rbind(
  df_metrics_a |>
    mutate(study = "study_a"),
  df_metrics_b |>
    mutate(study = "study_b"),
  df_metrics_c |>
    mutate(study = "study_c"),
  df_metrics_d |>
    mutate(study = "study_d")
) |>
  mutate(
    diff_rmsd = npe_rmsd - mcmc_rmsd,
    diff_pc = -(npe_pc - mcmc_pc),
    diff_ce = npe_ce - mcmc_ce
  ) |>
  pivot_longer(starts_with("diff"), names_to = c("metric"), names_prefix = "diff_", values_to = "diff") |>
  mutate(
    metric = fct_relabel(factor(metric, levels = c("rmsd", "pc", "ce")), toupper)
  )

df_join <- df_robustness |>
  group_by(study, drift_slope_loc, threshold_scale) |>
  summarize(mmd = median(sqrt(mmd))) |>
  left_join(df_metrics, by = c("study", "drift_slope_loc", "threshold_scale"))

df_join |>
  filter(metric == "RMSD") |>
  ggplot(aes(x = mmd, y = diff, color = study)) +
  facet_wrap(
    vars(param),
    labeller = label_parsed
  ) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "grey") +
  labs(x = "Posterior mismatch (median MMD)", y = "Difference RMSD (NPE - MCMC)", color = "Context-aware") +
  scale_x_continuous(breaks = seq(0.1, 1.3, 0.3), limits = c(0.05, 1.3)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open() +
  theme(
    strip.background = element_rect(color = NA, fill = "lightgrey"),
    legend.position = c(0.85, 0.2),
    legend.direction = "vertical",
    legend.justification = "center"
  )

ggsave(file.path(figure_path, "study_2_rmsd.png"))

df_join |>
  filter(metric == "PC") |>
  ggplot(aes(x = mmd, y = diff, color = study)) +
  facet_wrap(
    vars(param),
    labeller = label_parsed
  ) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "grey") +
  labs(x = "Posterior mismatch (median MMD)", y = "Difference PC (NPE - MCMC)", color = "Context-aware") +
  scale_x_continuous(breaks = seq(0.1, 1.3, 0.3), limits = c(0.05, 1.3)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open() +
  theme(
    strip.background = element_rect(color = NA, fill = "lightgrey"),
    legend.position = c(0.85, 0.2),
    legend.direction = "vertical",
    legend.justification = "center"
  )

ggsave(file.path(figure_path, "appendix_study_2_pc.png"))

df_join |>
  filter(metric == "CE") |>
  ggplot(aes(x = mmd, y = diff, color = study)) +
  facet_wrap(
    vars(param),
    labeller = label_parsed
  ) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "grey") +
  labs(x = "Posterior mismatch (median MMD)", y = "Difference CE (NPE - MCMC)", color = "Context-aware") +
  scale_x_continuous(breaks = seq(0.1, 1.3, 0.3), limits = c(0.05, 1.3)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open() +
  theme(
    strip.background = element_rect(color = NA, fill = "lightgrey"),
    legend.position = c(0.85, 0.2),
    legend.direction = "vertical",
    legend.justification = "center"
  )

ggsave(file.path(figure_path, "appendix_study_2_ce.png"))
