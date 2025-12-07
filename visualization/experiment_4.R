

library(ggplot2)
library(ggh4x)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"

data_path <- "outputs/experiment_4"

tasks <- c("FF1", "FN1", "FV1")
task_labels <- c("Figural", "Numeric", "Verbal")

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

# Empirical data distributions --------------------------------------------

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
  mutate(name = factor(name, tasks, task_labels)) |>
  group_by(name, pp) |>
  reframe(
    quantile_df(rt)
  ) |>
  ggplot(aes(x = as.factor(quant), y = val)) +
  facet_grid(cols = vars(name)) +
  geom_boxplot(show.legend = FALSE) +
  labs(x = "", y = "Response time") +
  scale_x_discrete(labels = c("Min", "1st", "Med", "3rd", "Max")) +
  theme_half_open()

plot_grid(p1, p2, labels = c("A", "B"), ncol = 1)

ggsave(file.path(figure_path, "appendix_study_4_descriptive.png"))


# Posterior predictive ----------------------------------------------------

df_ppd <- Reduce(rbind, lapply(models, function(mod) {
  read.csv(file.path(data_path, mod, "flow_matching/posterior_predictive/ppd.csv"))[,-1] |>
    mutate(study = mod, name = name_mcmc)
}))

df_ppd_acc <- df_ppd |>
  pivot_longer(cols = c(acc_est_mcmc, acc_est_npe), names_to = c("method"), values_to = "acc_est", names_pattern = "^acc_(?:est_)?(mcmc|npe|npe_no_params)$") |>
  # mutate(acc_true = if_else(method == "npe_no_params", acc_true_no_params, acc_true)) |>
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
    "npe" ~ "NPE"
  ),
    study = factor(study, levels = models, labels = study_labels),
    task = factor(name, levels = tasks, labels = task_labels)
  )

df_ppd_rt <- df_ppd |>
  pivot_longer(cols = c(rt_est_mcmc, rt_est_npe), names_to = c("measure", "method"), values_to = "value", names_pattern = "^(rt|acc)_(?:est_)?(mcmc|npe|npe_no_params)$") |>
  # mutate(rt_true = if_else(method == "npe_no_params", rt_true_no_params, rt_true)) |>
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
    "npe" ~ "NPE (context-aware)"
  ),
    study = factor(study, levels = models, labels = study_labels),
    task = factor(name, levels = tasks, labels = task_labels)
  )

df_ppd_acc |>
  ggplot(aes(x = acc_true, color = method)) +
  facet_grid2(rows = vars(study), cols = vars(task), scales = "free_x") +
  geom_pointrange(
    aes(
      y = acc_est_median,
      ymin = acc_est_lower,
      ymax = acc_est_upper
    ),
    size = 0.2,
    alpha = 0.5,
    position = position_jitter(width = 0.005)
  ) +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "True accuracy", y = "Posterior predictive accuracy", color = "") +
  # scale_color_viridis_d() +
  scale_color_manual(values = c("indianred", "lightblue")) +
  theme_half_open() +
  theme(
    legend.position = "top",
    legend.justification = "center",
    axis.text = element_text(size = 10),
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_4_ppd_acc_recovery.png"), width = 10, height = 6)

df_ppd_acc |>
  ggplot(aes(x = "", y = acc_rmsd, color = method)) +
  facet_grid(cols = vars(task), rows = vars(study)) +
  geom_boxplot(outlier.size = 1) +
  geom_hline(yintercept = 0, color = "grey") +
  labs(x = "", y = "Accuracy RMSD", color = "") +
  # scale_color_viridis_d() +
  # scale_y_continuous(limits = c(0, 0.2), breaks = c(0, 0.1, 0.2)) +
  scale_color_manual(values = c("indianred", "lightblue")) +
  theme_half_open() +
  theme(
    axis.ticks.x = element_blank(),
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_4_ppd_acc_rmsd.png"), width = 10, height = 6)

df_ppd_rt |>
  filter(quantile %in% c(0.1, 0.5, 0.9)) |>
  ggplot(aes(x = rt_true, color = method)) +
  facet_grid(rows = vars(study), cols = vars(task), scales = "free_x") +
  geom_pointrange(
    aes(
      y = rt_median,
      ymin = rt_lower,
      ymax = rt_upper
    ),
    size = 0.2,
    alpha = 0.5
  ) +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "True response time", y = "Posterior predictive response time", color = "") +
  # scale_color_viridis_d() +
  scale_color_manual(values = c("indianred", "lightblue")) +
  theme_half_open()  +
  theme(
    legend.position = "top",
    legend.justification = "center",
    axis.text = element_text(size = 10),
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_4_ppd_rt_recovery.png"), width = 10, height = 6)

df_ppd_rt |>
  filter(quantile %in% c(0.1, 0.5, 0.9)) |>
  ggplot(aes(x = as.factor(quantile), y = rt_rmsd, color = method)) +
  facet_grid(cols = vars(task), rows = vars(study)) +
  geom_boxplot() +
  geom_hline(yintercept = 0, color = "grey") +
  # scale_color_viridis_d() +
  scale_color_manual(values = c("indianred", "lightblue")) +
  labs(x = "Quantile", y = "Response time RMSD", color = "") +
  theme_half_open() +
  theme(
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_4_ppd_rt_rmsd.png"), width = 10, height = 6)


# Posterior mismatch ------------------------------------------------------

df_summary <- Reduce(rbind, lapply(models, function(mod) {
  read.csv(file.path(data_path, mod, "flow_matching/metrics/summary_stats.csv"))[,-1] |>
    mutate(study = mod)
}))

df_prior <- Reduce(rbind, lapply(models, function(mod) {
  read.csv(file.path("outputs/experiment_2", mod, "flow_matching/metrics/prior_stats.csv"))[,-1] |>
    mutate(study = mod)
}))

df_segment_prior <- df_prior |>
  pivot_longer(cols = c(v_intercept, v_slope, s_true, b, t0), names_to = "param") |>
  group_by(study, param) |>
  summarize(
    x = quantile(value, 0.001),
    xend = quantile(value, 0.999)
  ) |>
  mutate(
    study = factor(study, levels = models, labels = study_labels)
  )

df_range <- data.frame(
  param = rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2),
  mcmc_median = c(0.5, 4, 0.5, 2.5, 0.1, 0.9, 0, 2.5, 1, 5),
  npe_median = c(0.5, 4, 0.5, 2.5, 0.1, 0.9, 0, 2.5, 1, 5),
  study = factor(rep(models, each=10), levels = models, labels = study_labels)
)

df_summary |>
  mutate(
    study = factor(study, levels = models, labels = study_labels)
  ) |>
  ggplot(aes(x = mcmc_median, y = npe_median, color = study)) +
  facet_grid2(rows = vars(study), cols = vars(param), "free", independent = "y") +
  geom_point() +
  geom_rect(
    data = df_segment_prior,
    aes(xmin = x, xmax = xend, ymin = 0, ymax = Inf),
    inherit.aes = FALSE,
    fill = "grey",
    alpha = 0.3
  ) +
  geom_blank(data = df_range) +
  geom_abline(slope = 1, intercept = 0) +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "MCMC posterior median", y = "NPE posterior median") +
  theme_half_open() +
  theme(
    legend.position = "none",
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_4_recovery.png"), width = 10, height = 6)

df_range <- data.frame(
  param = rep(rep(c("b", "s_true", "t0", "v_intercept", "v_slope"), each = 2), 6),
  error_rate = 0,
  median_diff = rep(c(0, 1.5, 0, 1.0, 0, 0.3, 0, 1.0, 0, 2), 6),
  study = factor(rep(models, each=10))
) |>
  mutate(
    study = factor(study, levels = models, labels = study_labels)
  )

df_segment_error_rate <- df_prior |>
  group_by(study) |>
  summarize(
    x = 100 * quantile(acc, 0.001),
    xend = min(100 * quantile(acc, 0.999), 25)
  ) |>
  mutate(study = factor(study, levels = models, labels = study_labels))

df_summary |>
  mutate(
    study = factor(study, levels = models, labels = study_labels),
    median_diff = abs(mcmc_median - npe_median),
    error_rate = (1-acc)*100
  ) |>
  ggplot(aes(x = error_rate, y = median_diff, color = study)) +
  facet_grid2(rows = vars(study), cols = vars(param), scales = "free_y", independent = "y") +
  geom_point(alpha = 0.1)  +
  geom_rect(
    data = df_segment_error_rate,
    aes(xmin = x, xmax = xend, ymin = 0, ymax = Inf),
    inherit.aes = FALSE,
    fill = "grey",
    alpha = 0.3
  ) +
  geom_blank(data = df_range) +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "Error rate (in %)", y = "Absolute difference posterior median") +
  theme_half_open() +
  theme(
    legend.position = "none",
    strip.text.y = element_text(angle=360, hjust = 0),
    strip.background.y = element_blank()
  )

ggsave(file.path(figure_path, "study_4_posterior_mismatch.png"), width = 10, height = 6)
