
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
  pivot_longer(cols = c(RMSD, PC, CE), names_to = "metric") |>
  mutate(is_best = number == 12)

df_trials |>
  ggplot(aes(x = number, y = value, color = is_best)) +
  facet_grid(cols = vars(metric)) +
  geom_point(show.legend = FALSE) +
  labs(x = "Optuna trial number", y = "") +
  scale_color_manual(values = c("black", "red")) +
  theme_half_open()

ggsave(file.path(figure_path, "appendix_optimization.png"))
