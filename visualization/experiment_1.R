
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

data_path <- "outputs/experiment_1"

df_robustness_a <- read.csv(file.path(data_path, "rdm_simple/flow_matching/robustness/mmd.csv"))[,-1] # Skip first index column

df_robustness_b <- read.csv(file.path(data_path, "rdm_simple_discrete_lower/flow_matching/robustness/mmd.csv"))[,-1]

df_robustness_c <- read.csv(file.path(data_path, "rdm_simple_discrete_upper/flow_matching/robustness/mmd.csv"))[,-1]

df_robustness_d <- read.csv(file.path(data_path, "rdm_simple_discrete_full/flow_matching/robustness/mmd.csv"))[,-1]

df_summary_a <- read.csv(file.path(data_path, "rdm_simple/flow_matching/metrics/summary_stats.csv"))[,-1]

df_summary_b <- read.csv(file.path(data_path, "rdm_simple_discrete_lower/flow_matching/metrics/summary_stats.csv"))[,-1]

df_summary_c <- read.csv(file.path(data_path, "rdm_simple_discrete_upper/flow_matching/metrics/summary_stats.csv"))[,-1]

df_summary_d <- read.csv(file.path(data_path, "rdm_simple_discrete_full/flow_matching/metrics/summary_stats.csv"))[,-1]

join_robustness_summary_dfs <- function(df_summary, df_robustness) {
  return(df_summary |>
           group_by(sample_size, param) |>
           mutate(id = row_number(), .before = 1) |>
           left_join(df_robustness |> 
                       group_by(sample_size) |> 
                       mutate(id = row_number()), 
                     by=c("sample_size", "id")))
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
  true = c(0, 3, 0, 3, 0, 1, 0, 4, 0, 4),
  value = c(0, 3, 0, 3, 0, 1, 0, 4, 0, 4),
  method = "MCMC"
)

df_join |>
  pivot_longer(c(mcmc_median, npe_median), names_to = "method") |>
  mutate(
    study = factor(study, labels = study_labels),
    method = case_match(method, "mcmc_median" ~ "MCMC", .default = "NPE")
  ) |>
  ggplot(aes(x = true, y = value, color = method)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free", independent = "y") +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  geom_blank(data = df_range) +
  labs(x = "True parameter", y = "Posterior median", color = "") +
  theme_half_open() +
  theme(
    legend.position = "top",
    legend.justification = "center",
    strip.text.y = element_text(angle=360),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_1_recovery.png"))

df_range <- data.frame(
  param = rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2),
  sample_size = rep(c(50, 1250), each = 5),
  median_diff = c(0, 0.08, 0., 0.06, 0, 0.03, 0., 0.2, 0, 0.15),
  study = "study_a"
)

df_join |>
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
    color = "Trial number (training datasets)"
  ) +
  scale_x_continuous(breaks = c(50, 250, 500, 750, 1000, 1200), limits = c(50, 1200)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open()

ggsave(file.path(figure_path, "study_1_posterior_mismatch.png"))


# Metrics -----------------------------------------------------------------

df_metrics_a <- read.csv("outputs/experiment_1/rdm_simple/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_b <- read.csv("outputs/experiment_1/rdm_simple_discrete_lower/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_c <- read.csv("outputs/experiment_1/rdm_simple_discrete_upper/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_d <- read.csv("outputs/experiment_1/rdm_simple_discrete_full/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

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
  group_by(study, sample_size) |>
  summarize(mmd = median(sqrt(mmd))) |>
  left_join(df_metrics, by = c("study", "sample_size"))

df_join |>
  filter(metric == "RMSD") |>
  ggplot(aes(x = mmd, y = diff, color = study)) +
  facet_wrap(
    vars(param),
    labeller = label_parsed
  ) +
  geom_hline(yintercept = 0, color = "grey") +
  geom_point(alpha = 0.5) +
  # geom_smooth(mapping = aes(x = mmd, y = diff), inherit.aes = FALSE, method= "lm", color = "black") +
  # geom_density2d() +
  # geom_smooth(method = "lm", data = df_join |> filter(sample_size != 50)) +
  # geom_vline(xintercept = c(250, 500, 750, 1000), alpha = 0.1) +
  labs(x = "Posterior mismatch (median MMD)", y = "Difference RMSD (NPE - MCMC)", color = "Trial number (training datasets)") +
  scale_x_continuous(breaks = seq(0.1, 0.4, 0.1), limits = c(0.08, 0.4)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open() +
  theme(
    strip.background = element_rect(color = NA, fill = "lightgrey"),
    # strip.placement = "outside",
    legend.position = c(0.85, 0.2),
    legend.direction = "vertical",
    legend.justification = "center"
  )

ggsave(file.path(figure_path, "study_1_rmsd.png"))

df_join |>
  filter(metric == "PC") |>
  ggplot(aes(x = mmd, y = diff, color = study)) +
  facet_wrap(
    vars(param),
    labeller = label_parsed
  ) +
  geom_hline(yintercept = 0, color = "grey") +
  geom_point(alpha = 0.5) +
  # geom_smooth(mapping = aes(x = mmd, y = diff), inherit.aes = FALSE, method= "lm", color = "black") +
  # geom_density2d() +
  # geom_smooth(method = "lm", data = df_join |> filter(sample_size != 50)) +
  # geom_vline(xintercept = c(250, 500, 750, 1000), alpha = 0.1) +
  labs(x = "Posterior mismatch (median MMD)", y = "Difference PC (NPE - MCMC)", color = "Trial number (training datasets)") +
  scale_x_continuous(breaks = seq(0.1, 0.4, 0.1), limits = c(0.08, 0.4)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open() +
  theme(
    strip.background = element_rect(color = NA, fill = "lightgrey"),
    # strip.placement = "outside",
    legend.position = c(0.85, 0.2),
    legend.direction = "vertical",
    legend.justification = "center"
  )

ggsave(file.path(figure_path, "appendix_study_1_pc.png"))

df_join |>
  filter(metric == "CE") |>
  ggplot(aes(x = mmd, y = diff, color = study)) +
  facet_wrap(
    vars(param),
    labeller = label_parsed
  ) +
  geom_hline(yintercept = 0, color = "grey") +
  geom_point(alpha = 0.5) +
  # geom_smooth(mapping = aes(x = mmd, y = diff), inherit.aes = FALSE, method= "lm", color = "black") +
  # geom_density2d() +
  # geom_smooth(method = "lm", data = df_join |> filter(sample_size != 50)) +
  # geom_vline(xintercept = c(250, 500, 750, 1000), alpha = 0.1) +
  labs(x = "Posterior mismatch (median MMD)", y = "Difference CE (NPE - MCMC)", color = "Trial number (training datasets)") +
  scale_x_continuous(breaks = seq(0.1, 0.4, 0.1), limits = c(0.08, 0.4)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open() +
  theme(
    strip.background = element_rect(color = NA, fill = "lightgrey"),
    # strip.placement = "outside",
    legend.position = c(0.85, 0.2),
    legend.direction = "vertical",
    legend.justification = "center"
  )

ggsave(file.path(figure_path, "appendix_study_1_ce.png"))
