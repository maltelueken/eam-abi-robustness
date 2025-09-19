
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"


# Generalization ----------------------------------------------------------

df_robustness_a <- read.csv("outputs/experiment_1/rdm_simple/flow_matching/robustness/mmd.csv")[,-1] # Skip first index column

df_robustness_b <- read.csv("outputs/experiment_1/rdm_simple_discrete_lower/flow_matching/robustness/mmd.csv")[,-1] # Skip first index column

df_robustness_c <- read.csv("outputs/experiment_1/rdm_simple_discrete_upper/flow_matching/robustness/mmd.csv")[,-1] # Skip first index column

df_robustness_d <- read.csv("outputs/experiment_1/rdm_simple_discrete_full/flow_matching/robustness/mmd.csv")[,-1] # Skip first index column

df_robustness <- rbind(
  df_robustness_a |> 
    mutate(
      study = "study_a",
      context = case_when(
        sample_size %in% c(250, 500, 750, 1000) ~ "Training",
        sample_size > 1000 | sample_size < 250 ~ "Extrapolation",
        .default = "Interpolation"
      )
    ),
  df_robustness_b |> 
    mutate(
      study = "study_b",
      context = case_when(
        sample_size <= 250 ~ "Training",
        .default = "Extrapolation"
      )
    ),
  df_robustness_c |> 
    mutate(
      study = "study_c",
      context = case_when(
        sample_size >= 1000 ~ "Training",
        .default = "Extrapolation"
      )
    ),
  df_robustness_d |> 
    mutate(
      study = "study_d",
      context = "Training"
    )
) |>
  mutate(mmd_sqrt = sqrt(mmd))

study_labels <- c(
  TeX("A: 250, 500, 750, 1000"),
  TeX("B: 50, 100, ..., 250"),
  TeX("C: 1000, 1050, ..., 1200"),
  TeX("D: 50, 100, ..., 1200")
)

p1 <- df_robustness |> 
  group_by(sample_size, study) |>
  summarise(mmd = median(mmd_sqrt),
            lower = quantile(mmd_sqrt, probs = c(0.25)),
            upper = quantile(mmd_sqrt, probs = c(0.75))) |>
  # filter(study == "study_b") |>
  ggplot(aes(x = sample_size, y = mmd, color = study)) +
  # geom_ribbon(aes(x = as.numeric(sample_size), ymin = lower, ymax = upper, fill = study), alpha = 0.05, show.legend = FALSE) +
  geom_line(aes(group = study)) +
  geom_point() +
  geom_vline(xintercept = c(250, 500, 750, 1000), alpha = 0.1) +
  labs(x = "Trial number (test datasets)", y = "Posterior mismatch (MMD)", color = "Trial number (training datasets)") +
  scale_x_continuous(breaks = c(50, 250, 500, 750, 1000, 1200)) +
  scale_y_continuous(breaks = seq(0, 0.4, 0.1), limits = c(0, 0.4)) +
  scale_color_discrete(labels = study_labels) +
  theme_half_open() +
  theme(legend.justification = "center")

p2 <- df_robustness |> 
  ggplot(aes(x = context, y = sqrt(mmd), fill = study)) +
  geom_boxplot() +
  labs(x = "Setting", y = "", fill = "Study") +
  scale_fill_discrete(labels = study_labels) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1)) +
  theme_half_open()
  # theme(legend.position = "top", legend.direction = "vertical")

plot_grid(
  get_legend(p1),
  plot_grid(
    p1 + theme(legend.position = "none"),
    p2 + theme(legend.position = "none"),
    labels = "AUTO"
  ),
  ncol = 1,
  rel_heights = c(0.3, 1.0)
)

ggsave(file.path(figure_path, "study_1_posterior_mmd.png"))


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
