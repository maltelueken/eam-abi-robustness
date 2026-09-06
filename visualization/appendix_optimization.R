
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(latex2exp)
library(cowplot)

figure_path <- "visualization/figures/"

# One sweep per model family, so one trials file per family. `is_best` is written by
# scripts/select_architecture.py, which is also what picks the architecture the experiments then
# train at -- the figure marks the trial that was actually used rather than a hardcoded number,
# which silently became wrong the first time the sweep was rerun.
families <- c(rdm = "RDM", lba = "LBA")

df_trials <- bind_rows(lapply(names(families), function(family) {
  read.csv(file.path("multirun", paste0("trials_", family, ".csv"))) |>
    mutate(family = families[[family]])
})) |>
  filter(state == "COMPLETE") |>
  mutate(is_best = as.logical(is_best), family = fct_relevel(family, unname(families))) |>
  rename(RMSD = values_0, PC = values_1, CE = values_2) |>
  pivot_longer(cols = c(RMSD, PC, CE), names_to = "metric")

df_trials |>
  ggplot(aes(x = number, y = value, color = is_best)) +
  facet_grid(rows = vars(family), cols = vars(metric), scales = "free_y") +
  geom_point(show.legend = FALSE) +
  labs(x = "Optuna trial number", y = "") +
  scale_color_manual(values = c("black", "red")) +
  theme_half_open() +
  panel_border()

ggsave(file.path(figure_path, "appendix_optimization.png"))
