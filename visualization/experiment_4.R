

library(ggplot2)
library(ggh4x)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"

data_path <- "outputs/experiment_4"

# Empirical data distributions --------------------------------------------

task_labels <- c("Figural", "Numeric", "Verbal")

data_fn1 <- read.delim("outputs/experiment_4/rdm_simple/test_data/FN1.txt", sep = " ") |>
  mutate(name = "FN1")
data_fv1 <- read.delim("outputs/experiment_4/rdm_simple/test_data/FV1.txt", sep = " ") |>
  mutate(name = "FV1")
data_ff1 <- read.delim("outputs/experiment_4/rdm_simple/test_data/FF1.txt", sep = " ") |>
  mutate(name = "FF1")

quantile_df <- function(x, probs = seq(0, 1, 0.25)) {
  tibble(
    val = quantile(x, probs, na.rm = TRUE),
    quant = probs
  )
}

data <- rbind(data_fn1, data_fv1, data_ff1) |>
  filter(block == "test") |>
  mutate(rt = RT / 1000)

p1 <- data |>
  group_by(name, pp) |>
  summarise(acc = mean(acc)) |>
  ggplot(aes(x = name, y = acc)) +
  geom_boxplot() +
  labs(x = "Task", y = "Accuracy") +
  scale_x_discrete(labels = task_labels) +
  theme_half_open()

p2 <- data |>
  group_by(name, pp) |>
  reframe(
    quantile_df(rt)
  ) |>
  ggplot(aes(x = name, y = val, group = interaction(name, quant))) +
  geom_boxplot(show.legend = FALSE) +
  labs(x = "Task", y = "Response time") +
  scale_x_discrete(labels = task_labels) +
  theme_half_open()

plot_grid(p1, p2, labels = c("A", "B"))

ggsave(file.path(figure_path, "appendix_study_4_descriptive.png"))


# Robustness --------------------------------------------------------------

study_labels <- paste("Context", c("A:\nHigh error rate", "B:\nLow error rate", "C:\nHigh + low\nerror rate"))

df_robustness_a <- read.csv("outputs/experiment_4/rdm_simple/flow_matching/robustness/mmd.csv")[,-1] # Skip first index column

df_robustness_b <- read.csv("outputs/experiment_4/rdm_simple_meta_lower/flow_matching/robustness/mmd.csv")[,-1] # Skip first index column

df_robustness_c <- read.csv("outputs/experiment_4/rdm_simple_meta_upper/flow_matching/robustness/mmd.csv")[,-1] # Skip first index column

df_robustness_d <- read.csv("outputs/experiment_4/rdm_simple_meta/flow_matching/robustness/mmd.csv")[,-1] # Skip first index column

df_robustness <- rbind(
  df_robustness_a |>
    mutate(study = "study_a"),
  df_robustness_b |>
    mutate(study = "study_b"),
  df_robustness_c |>
    mutate(study = "study_c"),
  df_robustness_d |>
    mutate(study = "study_d")
) |>
  mutate(
    mmd_sqrt = sqrt(mmd),
    name = case_match(
      name,
      "FN1" ~ "Numeric",
      "FV1" ~ "Verbal",
      "FF1" ~ "Figural"))

df_robustness |>
  ggplot(aes(x = study, y = mmd_sqrt, fill = study)) +
  facet_wrap(vars(name)) +
  geom_boxplot() +
  labs(x = "Context-aware", y = "Maximum mean discrepancy") +
  scale_x_discrete(labels = study_labels) +
  scale_y_continuous(limits = c(0, 2), breaks = seq(0, 2, 0.5)) +
  scale_fill_discrete(guide = FALSE) +
  theme_half_open() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(figure_path, "study_4_posterior_mmd.png"))


# Metrics -----------------------------------------------------------------

df_metrics_a <- read.csv("outputs/experiment_4/rdm_simple/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_b <- read.csv("outputs/experiment_4/rdm_simple_meta_lower/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_c <- read.csv("outputs/experiment_4/rdm_simple_meta_upper/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics_d <- read.csv("outputs/experiment_4/rdm_simple_meta/flow_matching/metrics/metrics.csv")[,-1] # Skip first index column

df_metrics <- rbind(
  df_metrics_a |>
    mutate(study = "study_a"),
  df_metrics_b |>
    mutate(study = "study_b"),
  df_metrics_c |>
    mutate(study = "study_c"),
  df_metrics_d |>
    mutate(study = "study_d")
)

df_metrics |>
  ggplot(aes(x = mcmc_median, y = npe_median, color = study)) +
  facet_grid(rows = vars(name), cols = vars(param), scales = "free") +
  geom_abline(slope = 1, intercept = 0) +
  geom_smooth(method = "lm") +
  geom_point()


# Posterior predictive ----------------------------------------------------

df_ppd_a <- read.csv(file.path(data_path, "rdm_simple/flow_matching/posterior_predictive/ppd.csv"))[,-1] # Skip first index column

df_ppd_b <- read.csv(file.path(data_path, "rdm_simple_meta_lower/flow_matching/posterior_predictive/ppd.csv"))[,-1]
df_ppd_b_no_params <- read.csv(file.path(data_path, "rdm_simple_meta_lower_no_params/flow_matching/posterior_predictive/ppd.csv"))[,-1]

df_ppd_c <- read.csv(file.path(data_path, "rdm_simple_meta_upper/flow_matching/posterior_predictive/ppd.csv"))[,-1]
df_ppd_c_no_params <- read.csv(file.path(data_path, "rdm_simple_meta_upper_no_params/flow_matching/posterior_predictive/ppd.csv"))[,-1]

df_ppd_d <- read.csv(file.path(data_path, "rdm_simple_meta/flow_matching/posterior_predictive/ppd.csv"))[,-1]
df_ppd_d_no_params <- read.csv(file.path(data_path, "rdm_simple_meta_no_params/flow_matching/posterior_predictive/ppd.csv"))[,-1]

join_ppd_dfs <- function(df_1, df_2) {
  return(df_1 |>
           left_join(
             df_2,
             by = c("name_mcmc", "id", "sample", "quantile"),
             suffix = c("", "_no_params"
             ))
  )
}

df_ppd <- rbind(
  join_ppd_dfs(df_ppd_b, df_ppd_b_no_params) |>
    mutate(study = "study_b"),
  join_ppd_dfs(df_ppd_c, df_ppd_c_no_params) |>
    mutate(study = "study_c"),
  join_ppd_dfs(df_ppd_d, df_ppd_d_no_params) |>
    mutate(study = "study_d")
) |>
  mutate(
    study = factor(study, labels = study_labels),
    name = factor(name_mcmc, labels = task_labels)
  )

df_ppd_acc <- df_ppd |>
  pivot_longer(cols = c(acc_est_mcmc, acc_est_npe, acc_est_npe_no_params), names_to = c("method"), values_to = "acc_est", names_pattern = "^acc_(?:est_)?(mcmc|npe|npe_no_params)$") |>
  mutate(acc_true = if_else(method == "npe_no_params", acc_true_no_params, acc_true)) |>
  group_by(study, name, id, method) |>
  summarise(
    acc_rmsd = sqrt(mean((acc_true - acc_est)^2)),
    acc_est_median = median(acc_est),
    acc_est_lower = quantile(acc_est, 0.025),
    acc_est_upper = quantile(acc_est, 0.975),
    acc_true = median(acc_true)
  ) |>
  mutate(method = case_match(
    method,
    "mcmc" ~ "MCMC",
    "npe" ~ "NPE (context-aware)",
    "npe_no_params" ~ "NPE (context-unaware)"
  ))

df_ppd_rt <- df_ppd |>
  pivot_longer(cols = c(rt_est_mcmc, rt_est_npe, rt_est_npe_no_params), names_to = c("measure", "method"), values_to = "value", names_pattern = "^(rt|acc)_(?:est_)?(mcmc|npe|npe_no_params)$") |>
  mutate(rt_true = if_else(method == "npe_no_params", rt_true_no_params, rt_true)) |>
  pivot_wider(id_cols = c(study, name, id, sample, method, quantile, rt_true), names_from = measure, values_from = value) |>
  group_by(study, name, id, method, quantile) |>
  summarise(
    rt_rmsd = sqrt(mean((rt_true - rt)^2)),
    rt_median = median(rt),
    rt_lower = quantile(rt, 0.025),
    rt_upper = quantile(rt, 0.975),
    rt_true = median(rt_true)
  ) |>
  mutate(method = case_match(
    method,
    "mcmc" ~ "MCMC",
    "npe" ~ "NPE (context-aware)",
    "npe_no_params" ~ "NPE (context-unaware)"
  ))

p1 <- df_ppd_acc |>
  ggplot(aes(x = acc_true, color = method)) +
  facet_grid2(rows = vars(study), cols = vars(name), scales = "free_x") +
  geom_pointrange(
    aes(
      y = acc_est_median,
      ymin = acc_est_lower,
      ymax = acc_est_upper
    ),
    size = 0.1,
    position = position_jitter(width = 0.005)
  ) +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "True accuracy", y = "Posterior predictive accuracy", color = "") +
  scale_color_viridis_d() +
  theme_half_open() +
  theme(
    legend.position = "top",
    legend.justification = "center",
    axis.text = element_text(size = 10),
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

p2 <- df_ppd_acc |>
  ggplot(aes(x = "", y = acc_rmsd, color = method)) +
  facet_grid(cols = vars(name), rows = vars(study)) +
  geom_boxplot(outlier.size = 1) +
  labs(x = "", y = "Accuracy RMSD", color = "") +
  scale_color_viridis_d() +
  theme_half_open() +
  theme(
    axis.ticks.x = element_blank(),
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

plot_grid(
  get_legend(p1),
  plot_grid(
    p1 + theme(legend.position = "none"),
    p2 + theme(legend.position = "none", strip.text.y = element_blank()),
    ncol = 2,
    rel_widths = c(1.0, 0.4),
    labels = c("A", "B")
  ),
  rel_heights = c(0.1, 1.0),
  ncol = 1
)

ggsave(file.path(figure_path, "study_4_ppd_acc.png"), width = 10, height = 6)

p3 <- df_ppd_rt |>
  filter(quantile %in% c(0.1, 0.5, 0.9)) |>
  ggplot(aes(x = rt_true, color = method)) +
  facet_grid2(rows = vars(study), cols = vars(name), scales = "free", independent = "y") +
  geom_pointrange(
    aes(
      y = rt_median,
      ymin = rt_lower,
      ymax = rt_upper
    ),
    size = 0.2
  ) +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "True response time", y = "Posterior predictive response time", color = "") +
  scale_color_viridis_d() +
  theme_half_open()  +
  theme(
    legend.position = "top",
    legend.justification = "center",
    axis.text = element_text(size = 10),
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

p4 <- df_ppd_rt |>
  filter(quantile %in% c(0.1, 0.5, 0.9)) |>
  ggplot(aes(x = as.factor(quantile), y = rt_rmsd, color = method)) +
  facet_grid2(cols = vars(name), rows = vars(study), scales = "free_y", independent = "y") +
  geom_boxplot() +
  scale_color_viridis_d() +
  labs(x = "Quantile", y = "Response time RMSD", color = "") +
  theme_half_open() +
  theme(
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

plot_grid(
  get_legend(p3),
  plot_grid(
    p3 + theme(legend.position = "none"),
    p4 + theme(legend.position = "none", strip.text.x = element_blank()),
    ncol = 1,
    # rel_widths = c(1.0, 0.8),
    labels = c("A", "B")
  ),
  rel_heights = c(0.1, 1.0),
  ncol = 1
)

ggsave(file.path(figure_path, "study_4_ppd_rt.png"), width = 10, height = 8)


# Posterior mismatch ------------------------------------------------------

df_summary_c <- read.csv(file.path(data_path, "rdm_simple_meta/flow_matching/metrics/summary_stats.csv"))[,-1]
df_summary_c_no_params <- read.csv(file.path(data_path, "rdm_simple_meta_no_params/flow_matching/metrics/summary_stats.csv"))[,-1]

df_summary_a <- read.csv(file.path(data_path, "rdm_simple_meta_lower/flow_matching/metrics/summary_stats.csv"))[,-1]
df_summary_a_no_params <- read.csv(file.path(data_path, "rdm_simple_meta_lower_no_params/flow_matching/metrics/summary_stats.csv"))[,-1]

df_summary_b <- read.csv(file.path(data_path, "rdm_simple_meta_upper/flow_matching/metrics/summary_stats.csv"))[,-1]
df_summary_b_no_params <- read.csv(file.path(data_path, "rdm_simple_meta_upper_no_params/flow_matching/metrics/summary_stats.csv"))[,-1]

join_summary_dfs <- function(df_1, df_2) {
  return(df_1 |>
           group_by(name, param) |>
           mutate(id = row_number()) |>
           left_join(
             df_2 |>
               group_by(name, param) |>
               mutate(id = row_number()),
             by = c("name", "id", "param"),
             suffix = c("", "_no_params"
             ))
  )
}

df_summary <- rbind(
  join_summary_dfs(df_summary_a, df_summary_a_no_params) |> mutate(study = "study_a"),
  join_summary_dfs(df_summary_b, df_summary_b_no_params) |> mutate(study = "study_b"),
  join_summary_dfs(df_summary_c, df_summary_c_no_params) |> mutate(study = "study_c")
) |>
  pivot_longer(c(npe_median, npe_median_no_params), names_to = "method", values_to = "npe_median") |>
  mutate(
    acc = if_else(method == "npe_no_params", acc_no_params, acc),
    study = factor(study, labels = study_labels),
    name = factor(name, levels = c("FF1", "FN1", "FV1"), labels = task_labels),
    method = case_match(
      method,
      "npe_median" ~ "NPE (context-aware)",
      "npe_median_no_params" ~ "NPE (context-unaware)"
    )
  )

df_summary |>
  # filter(!param %in% c("t0", "s_true")) |>
  ggplot(aes(x = mcmc_median, y = npe_median, color = method)) +
  facet_grid2(rows = vars(study), cols = vars(param), "free", independent = "y") +
  geom_point() +
  geom_abline(slope = 1, intercept = 0)

df_summary |>
  group_by(study, param, method) |>
  summarise(r = cor(mcmc_median, npe_median)) |>
  group_by(study, method) |>
  summarise(mean(r))

df_summary |>
  ggplot(aes(x = 1-acc, y = abs(mcmc_median - npe_median), color = method)) +
  facet_grid2(rows = vars(study), cols = vars(param), "free_y") +
  geom_point() +
  geom_smooth(method = "lm")
