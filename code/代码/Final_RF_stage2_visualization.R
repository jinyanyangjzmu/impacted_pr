# ============================================================
# Impacted stone study
# Stage-2 visualization for the locked random-forest model
#
# This script reads the saved final-model bundle only. It does not refit
# LASSO, retune the random forest, or access external-validation data.
#
# Outputs:
#   Figure_final_RF_construction.pdf
#   Figure_final_RF_construction.png
#   final_RF_visualization_audit.csv
#   final_RF_tuning_boundary_audit.csv
#
# Usage:
# Rscript Final_RF_stage2_visualization.R [model_bundle.rds] [output_dir]
# ============================================================

# 1. Packages and paths --------------------------------------
required_pkgs <- c("ggplot2", "dplyr", "tibble", "patchwork", "scales")
missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_pkgs)) {
  stop("Install required packages first: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tibble)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
default_dir <- "C:/Users/Lenovo/Desktop/impacted_pr/output/final_selected_model"
bundle_path <- if (length(args) >= 1L) {
  args[[1]]
} else {
  file.path(default_dir, "final_random_forest_model_bundle.rds")
}
output_dir <- if (length(args) >= 2L) args[[2]] else dirname(bundle_path)

if (!file.exists(bundle_path)) stop("Model bundle not found: ", bundle_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 2. Read and validate bundle --------------------------------
bundle <- readRDS(bundle_path)
required_objects <- c(
  "selected_model", "selected_predictors", "selected_predictor_table",
  "lasso_summary", "lasso_cv_fit", "best_parameters",
  "tuning_grid", "tuning_metrics", "configuration"
)
missing_objects <- setdiff(required_objects, names(bundle))
if (length(missing_objects)) {
  stop("Bundle is missing: ", paste(missing_objects, collapse = ", "))
}
if (!identical(bundle$selected_model, "random_forest")) {
  stop("This script expects a locked random_forest bundle.")
}

cv_fit <- bundle$lasso_cv_fit
if (!inherits(cv_fit, "cv.glmnet")) {
  stop("bundle$lasso_cv_fit is not a valid cv.glmnet object.")
}

config_value <- function(setting, default = NA_character_) {
  value <- bundle$configuration$value[
    bundle$configuration$setting == setting
  ]
  if (length(value) == 1L) value else default
}

font_family <- if (.Platform$OS.type == "windows") "Arial" else "sans"
blue <- "#2F6F92"
light_blue <- "#DCEAF4"
red <- "#C44E52"
dark <- "#25343D"
grey <- "#8A9297"

theme_publication <- function() {
  theme_classic(base_size = 10.5, base_family = font_family) +
    theme(
      plot.title = element_text(face = "bold", size = 11.5, colour = dark),
      plot.subtitle = element_text(size = 9.2, colour = "#4E5960"),
      axis.title = element_text(size = 10.2, colour = dark),
      axis.text = element_text(size = 9.1, colour = dark),
      legend.title = element_text(size = 9.0),
      legend.text = element_text(size = 8.7),
      plot.margin = margin(7, 9, 7, 7)
    )
}

# 3. Panel A: final LASSO cross-validation -------------------
lasso_curve <- tibble(
  lambda = as.numeric(cv_fit$lambda),
  log_lambda = log(lambda),
  mean_deviance = as.numeric(cv_fit$cvm),
  se = as.numeric(cv_fit$cvsd),
  n_nonzero = as.integer(cv_fit$nzero)
) |>
  mutate(
    lower = mean_deviance - se,
    upper = mean_deviance + se
  )

lambda_min <- as.numeric(cv_fit$lambda.min)
lambda_1se <- as.numeric(cv_fit$lambda.1se)
lambda_used <- as.numeric(bundle$lasso_summary$lambda_used[[1]])
lambda_rule <- as.character(bundle$lasso_summary$lambda_rule[[1]])
used_index <- which.min(abs(lasso_curve$lambda - lambda_used))

lambda_lines <- tibble(
  rule = factor(
    c("lambda.min", "lambda.1se"),
    levels = c("lambda.min", "lambda.1se")
  ),
  x = log(c(lambda_min, lambda_1se))
)

p_a <- ggplot(lasso_curve, aes(log_lambda, mean_deviance)) +
  geom_ribbon(
    aes(ymin = lower, ymax = upper),
    fill = light_blue, alpha = 0.78
  ) +
  geom_line(linewidth = 0.75, colour = dark) +
  geom_point(size = 1.25, colour = blue) +
  geom_vline(
    data = lambda_lines,
    aes(xintercept = x, colour = rule, linetype = rule),
    linewidth = 0.75, show.legend = TRUE
  ) +
  geom_point(
    data = lasso_curve[used_index, , drop = FALSE],
    shape = 21, size = 3.5, stroke = 1.0,
    fill = "white", colour = dark
  ) +
  scale_colour_manual(
    values = c("lambda.min" = blue, "lambda.1se" = red),
    name = NULL
  ) +
  scale_linetype_manual(
    values = c("lambda.min" = "dashed", "lambda.1se" = "dotted"),
    name = NULL
  ) +
  labs(
    title = "Final LASSO selection in the complete development cohort",
    subtitle = sprintf(
      "Rule: %s; lambda used = %.4g; selected predictors = %d",
      lambda_rule, lambda_used, length(bundle$selected_predictors)
    ),
    x = expression(log(lambda)),
    y = "Mean cross-validated binomial deviance"
  ) +
  theme_publication() +
  theme(legend.position = "top")

# 4. Panel B: outer-fold selection stability -----------------
stability <- bundle$selected_predictor_table |>
  as_tibble()

required_stability <- c(
  "predictor", "selected_outer_folds", "successful_lasso_folds",
  "selection_frequency"
)
if (!all(required_stability %in% names(stability))) {
  stop("selected_predictor_table lacks outer-fold stability columns.")
}
if (!nrow(stability) || anyNA(stability$selection_frequency)) {
  stop("Outer-fold LASSO selection frequencies are unavailable.")
}

pretty_name <- function(x) {
  tools::toTitleCase(gsub("_", " ", x, fixed = TRUE))
}

stability <- stability |>
  mutate(
    display_name = pretty_name(predictor),
    display_name = reorder(display_name, selection_frequency),
    frequency_label = sprintf(
      "%d/%d (%.0f%%)",
      selected_outer_folds,
      successful_lasso_folds,
      100 * selection_frequency
    )
  )

p_b <- ggplot(stability, aes(selection_frequency, display_name)) +
  geom_segment(
    aes(x = 0, xend = selection_frequency, yend = display_name),
    linewidth = 1.2, colour = light_blue
  ) +
  geom_point(size = 3.3, colour = blue) +
  geom_text(
    aes(label = frequency_label), hjust = -0.12,
    size = 3.0, family = font_family, colour = dark
  ) +
  scale_x_continuous(
    limits = c(0, 1.13),
    breaks = seq(0, 1, 0.25),
    labels = scales::label_percent(accuracy = 1),
    expand = c(0, 0)
  ) +
  labs(
    title = "Stability of the final predictors",
    subtitle = "LASSO selection across outer training sets",
    x = "Outer-fold selection frequency",
    y = NULL
  ) +
  theme_publication()

# 5. Panel C: final random-forest tuning ---------------------
tuning_auc <- bundle$tuning_metrics |>
  as_tibble() |>
  filter(.metric == "roc_auc", is.finite(mean))

required_tuning <- c("mtry", "min_n", "mean", ".config")
if (!all(required_tuning %in% names(tuning_auc)) || !nrow(tuning_auc)) {
  stop("Valid random-forest mtry/min_n tuning results were not found.")
}

best_parameters <- bundle$best_parameters |> as_tibble()
if (!all(c("mtry", "min_n") %in% names(best_parameters)) ||
    nrow(best_parameters) != 1L) {
  stop("Exactly one final mtry/min_n combination is required.")
}

best_config <- if (".config" %in% names(best_parameters)) {
  best_parameters$.config[[1]]
} else {
  tuning_auc |>
    filter(
      mtry == best_parameters$mtry[[1]],
      min_n == best_parameters$min_n[[1]]
    ) |>
    slice_head(n = 1L) |>
    pull(.config)
}
if (length(best_config) != 1L || is.na(best_config)) {
  stop("The selected tuning configuration could not be identified.")
}

tuning_auc <- tuning_auc |>
  mutate(
    is_best = .config == best_config,
    mtry_plot = factor(mtry, levels = sort(unique(mtry))),
    min_n_plot = factor(min_n, levels = sort(unique(min_n)))
  )

best_point <- tuning_auc |> filter(is_best)
if (nrow(best_point) != 1L) {
  stop("The selected tuning configuration is absent or duplicated.")
}

tuning_v <- config_value("tuning_v")
tuning_repeats <- config_value("tuning_repeats")

p_c <- ggplot(tuning_auc, aes(mtry_plot, min_n_plot)) +
  geom_point(aes(colour = mean), size = 4.2, alpha = 0.92) +
  geom_point(
    data = best_point,
    shape = 21, size = 6.2, stroke = 1.15,
    fill = NA, colour = red
  ) +
  scale_colour_viridis_c(
    option = "C", direction = 1, end = 0.88,
    name = "Mean CV\nROC AUC"
  ) +
  labs(
    title = "Final random-forest tuning landscape",
    subtitle = sprintf(
      "%s-fold CV x %s repeats; selected mtry = %s, min_n = %s",
      tuning_v, tuning_repeats,
      best_parameters$mtry[[1]], best_parameters$min_n[[1]]
    ),
    x = "Number of variables sampled (mtry)",
    y = "Minimum node size (min_n)"
  ) +
  theme_publication() +
  theme(legend.position = "right")

# 6. Combine and save ----------------------------------------
figure <- patchwork::wrap_plots(
  p_a,
  patchwork::wrap_plots(p_b, p_c, ncol = 2, widths = c(1.02, 1.18)),
  ncol = 1,
  heights = c(0.95, 1.12)
) +
  patchwork::plot_annotation(
    title = "Final random-forest model construction",
    subtitle = "Complete development cohort; external-validation data were not accessed",
    caption = paste(
      "The tuning results in panel C support final model construction and",
      "are not an unbiased validation estimate."
    ),
    tag_levels = "A",
    theme = theme(
      text = element_text(family = font_family, colour = dark),
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10.2, colour = "#4E5960"),
      plot.caption = element_text(size = 8.7, colour = "#59636A"),
      plot.tag = element_text(face = "bold", size = 12.5, colour = dark)
    )
  )

pdf_path <- file.path(output_dir, "Figure_final_RF_construction.pdf")
png_path <- file.path(output_dir, "Figure_final_RF_construction.png")
pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf

ggsave(
  pdf_path, figure,
  width = 11.5, height = 8.5, units = "in",
  device = pdf_device, bg = "white"
)
ggsave(
  png_path, figure,
  width = 11.5, height = 8.5, units = "in",
  dpi = 600, bg = "white"
)

# 7. Reproducibility audits ----------------------------------
grid <- bundle$tuning_grid |> as_tibble()
boundary_audit <- bind_rows(lapply(c("mtry", "min_n"), function(parameter) {
  values <- grid[[parameter]]
  selected_value <- best_parameters[[parameter]][[1]]
  tibble(
    parameter = parameter,
    grid_min = min(values, na.rm = TRUE),
    grid_max = max(values, na.rm = TRUE),
    selected_value = selected_value,
    selected_at_lower_boundary = isTRUE(all.equal(selected_value, min(values, na.rm = TRUE))),
    selected_at_upper_boundary = isTRUE(all.equal(selected_value, max(values, na.rm = TRUE)))
  )
}))

visualization_audit <- tibble(
  item = c(
    "bundle_file", "selected_model", "selected_predictors",
    "lambda_rule", "lambda_used", "tuning_candidates",
    "best_config", "best_mtry", "best_min_n",
    "external_data_accessed", "model_refitted"
  ),
  value = as.character(c(
    normalizePath(bundle_path, winslash = "/", mustWork = TRUE),
    bundle$selected_model,
    length(bundle$selected_predictors),
    lambda_rule,
    lambda_used,
    nrow(tuning_auc),
    best_config,
    best_parameters$mtry[[1]],
    best_parameters$min_n[[1]],
    "No",
    "No"
  ))
)

write.csv(
  visualization_audit,
  file.path(output_dir, "final_RF_visualization_audit.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)
write.csv(
  boundary_audit,
  file.path(output_dir, "final_RF_tuning_boundary_audit.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

message(
  "Stage-2 visualization completed without refitting.\n",
  "PDF: ", pdf_path, "\n",
  "PNG: ", png_path, "\n",
  "Selected RF parameters: mtry = ", best_parameters$mtry[[1]],
  "; min_n = ", best_parameters$min_n[[1]]
)
