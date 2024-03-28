library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


lynx_hares_reduced <- read_csv("lynx-hares_ode_forecast_windowed.csv") |>
  select(-Hare) |>
  pivot_wider(id_cols = "index", names_from = "time", values_from = "Lynx") |>
  select(-`0`)

write_csv(lynx_hares_reduced, "hares_ode_forecast_windowed_reduced.csv")
