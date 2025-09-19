
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"

df_trials <- read.csv("multirun/trials.csv")[,-1] |>
  filter(state == "COMPLETE") |>
  rename(RMSD = values_0, PC = values_1, CE = values_2) |>
  pivot_longer(cols = c(RMSD, PC, CE), names_to = "metric")

df_trials |>
  ggplot(aes(x = number, y = value)) +
  facet_grid(cols = vars(metric)) +
  geom_point() +
  labs(x = "Optuna trial number", y = "") +
  theme_half_open()

ggsave(file.path(figure_path, "appendix_optimization.png"))
