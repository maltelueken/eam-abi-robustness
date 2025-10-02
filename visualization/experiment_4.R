

library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"


# Empirical data distributions --------------------------------------------

task_labels <- c("Numeric", "Verbal", "Figural")

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

study_labels <- c("A: No", "B: High error rate", "C: Low error rate", "D: Full")

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

df_ppd_a <- read.csv("outputs/experiment_4/rdm_simple/flow_matching/posterior_predictive/ppd.csv") # Skip first index column

df_ppd_b <- read.csv("outputs/experiment_4/rdm_simple_meta_lower/flow_matching/posterior_predictive/ppd.csv")# Skip first index column

df_ppd_c <- read.csv("outputs/experiment_4/rdm_simple_meta_upper/flow_matching/posterior_predictive/ppd.csv") # Skip first index column

df_ppd_d <- read.csv("outputs/experiment_4/rdm_simple_meta/flow_matching/posterior_predictive/ppd.csv")[,-1] # Skip first index column


df_ppd_d |>
  rename(name = name_mcmc) |>
  pivot_longer(cols = c(acc_est_mcmc, acc_est_npe), names_to = c("method"), values_to = "acc_est", names_pattern = "^acc_(?:est_)?(mcmc|npe)$") |>
  group_by(name, id, method) |>
  summarise(
    acc_est_median = median(acc_est),
    acc_est_lower = quantile(acc_est, 0.025),
    acc_est_upper = quantile(acc_est, 0.975),
    acc_true = median(acc_true)
  ) |>
  ggplot(aes(x = acc_true, color = method)) +
  facet_grid(rows = vars(name)) +
  geom_pointrange(
    aes(
      y = acc_est_median,
      ymin = acc_est_lower,
      ymax = acc_est_upper
    ),
    position = position_jitter(width = 0.005)
  ) +
  geom_abline(slope = 1, intercept = 0) +
  scale_color_viridis_d() +
  # lims(x = c(0.9, 1.0), y = c(0.9, 1.0)) +
  theme_half_open()

df_ppd_d |>
  rename(name = name_mcmc) |>
  pivot_longer(cols = c(rt_est_mcmc, rt_est_npe), names_to = c("measure", "method"), values_to = "value", names_pattern = "^(rt|acc)_(?:est_)?(mcmc|npe)$") |>
  pivot_wider(id_cols = c(name, id, sample, method, quantile, rt_true), names_from = measure, values_from = value) |>
  group_by(name, id, method, quantile) |>
  summarise(
    rt_median = median(rt),
    rt_lower = quantile(rt, 0.025),
    rt_upper = quantile(rt, 0.975),
    rt_true = median(rt_true)
  ) |>
  filter(quantile %in% c(0.1, 0.5, 0.9)) |>
  ggplot(aes(x = rt_true, color = method)) +
  facet_grid2(rows = vars(name), cols = vars(quantile), scales = "free", independent = "y") +
  geom_pointrange(
    aes(
      y = rt_median,
      ymin = rt_lower,
      ymax = rt_upper
    ),
    position = position_jitter(width = 0.005)
  ) +
  geom_abline(slope = 1, intercept = 0) +
  # lims(x = c(0.9, 1.0), y = c(0.9, 1.0)) +
  # coord_cartesian(ylim = c(0, 5)) +
  scale_color_viridis_d() +
  theme_half_open()

df_ppd <- rbind(
  df_ppd_a |>
    mutate(study = "study_a"),
  df_ppd_b |>
    mutate(study = "study_b"),
  df_ppd_c |>
    mutate(study = "study_c"),
  df_ppd_d |>
    mutate(study = "study_d")
)

df_ppd_join <- df_ppd |>
  group_by(study, name) |>
  mutate(index = row_number()) |>
  left_join(df_robustness |>
              group_by(study, name) |>
              mutate(index = row_number())) |>
  filter(npe_rt_rmsd < 2)

p3 <- df_ppd_join |>
  ggplot(aes(x = name, y = npe_acc_rmsd - mcmc_acc_rmsd, fill = study)) +
  facet_wrap(vars("Accuracy")) +
  geom_boxplot() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Task", y = "Difference NPE - MCMC error", fill = "Context-aware") +
  scale_x_discrete(labels = task_labels) +
  scale_fill_discrete(labels = study_labels) +
  theme_half_open() +
  theme(legend.justification = "center")

p4 <- df_ppd_join |>
  ggplot(aes(x = name, y = npe_rt_rmsd - mcmc_rt_rmsd, fill = study)) +
  facet_wrap(vars("Response time")) +
  geom_boxplot() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Task", y = "", fill = "Context-aware") +
  scale_x_discrete(labels = task_labels) +
  scale_fill_discrete(labels = study_labels) +
  theme_half_open()

plot_grid(
  get_legend(p3),
  plot_grid(
    p3 + theme(legend.position = "none"),
    p4 + theme(legend.position = "none"),
    labels = "AUTO"
  ),
  ncol = 1,
  rel_heights = c(0.3, 1)
)

ggsave(file.path(figure_path, "study_4_ppd.png"))

df_ppd_join |>
  ggplot(aes(x = mmd_sqrt, y = npe_acc_rmsd - mcmc_acc_rmsd, color = study)) +
  facet_wrap(vars(name)) +
  geom_point() +
  geom_smooth(method = "lm")

df_ppd_join |>
  ggplot(aes(x = mmd_sqrt, y = npe_rt_rmsd - mcmc_rt_rmsd, color = study)) +
  facet_wrap(vars(name)) +
  geom_point() +
  geom_smooth(method = "lm")

df_ppd_join |>
  ggplot(aes(x = mcmc_acc_rmsd, y = npe_acc_rmsd)) +
  geom_point() +
  geom_smooth() +
  geom_abline(intercept = 0, slope = 1)

df_ppd_join |>
  ggplot(aes(x = mcmc_rt_rmsd, y = npe_rt_rmsd)) +
  geom_point() +
  geom_smooth() +
  geom_abline(intercept = 0, slope = 1)

df_ppd |>
  ggplot(aes(y = npe_acc_rmsd, x = study)) +
  geom_boxplot() +
  geom_point()

df_ppd |>
  filter(npe_rt_rmsd < 1) |>
  ggplot(aes(y = npe_rt_rmsd, x = study)) +
  geom_boxplot() +
  geom_point()

df_ppd |>
  filter(npe_rt_rmsd < 1) |>
  ggplot(aes(y = npe_rt_rmsd, x = npe_acc_rmsd, color = study)) +
  geom_point()

df_ppd |>
  filter(npe_rt_rmsd < 1) |>
  ggplot(aes(y = mcmc_rt_rmsd, x = mcmc_acc_rmsd, color = study)) +
  geom_point()

df_ppd |>
  ggplot(aes(y = npe_acc_rmsd, x = mcmc_acc_rmsd, color = study)) +
  geom_point()

df_ppd |>
  filter(npe_rt_rmsd < 1) |>
  ggplot(aes(y = npe_rt_rmsd, x = mcmc_rt_rmsd, color = study)) +
  geom_point()

cbind(df_robustness, df_ppd) |>
  select(npe_acc_rmsd, mcmc_acc_rmsd, mmd_sqrt) |>
  ggplot(aes(x = mmd_sqrt, y = npe_acc_rmsd - mcmc_acc_rmsd)) +
  geom_point() +
  geom_smooth(method = "lm") +
  scale_color_viridis_c()

cbind(df_robustness, df_ppd) |>
  select(npe_rt_rmsd, mcmc_rt_rmsd, mmd_sqrt) |>
  filter(npe_rt_rmsd < 1) |>
  ggplot(aes(x = mmd_sqrt, y = npe_rt_rmsd - mcmc_rt_rmsd)) +
  geom_point() +
  geom_smooth(method = "lm") +
  scale_color_viridis_c()
