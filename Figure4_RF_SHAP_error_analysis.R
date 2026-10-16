# Figure 4 | Locked RF SHAP explanations and external error analysis
#
# This script generates:
#   1. Figure4_RF_SHAP_error_analysis.pdf/.tiff/.png
#      A: global mean absolute SHAP values
#      B: SHAP beeswarm
#      C: representative false-positive waterfall
#      D: representative false-negative waterfall
#   2. Figure4_external_probability_distributions.pdf/.tiff
#      external TP/TN/FP/FN predicted-probability distributions
#   3. Auditable CSV summaries and a draft figure legend
#
# The original project paths are intentionally preserved.
# No model is fitted in this script. The saved locked RF bundle is used as-is.
# SHAP values are calculated on the predicted probability of an impacted stone.

# ---- Paths and reproducible settings -----------------------------------------
root <- "C:/Users/Lenovo/Desktop/impacted_pr"
bundle_path <- file.path(root, "output", "final_selected_model",
                         "final_random_forest_model_bundle.rds")
development_path <- file.path(root, "data", "impacted_clean.rds")
external_path <- file.path(root, "data", "data_wb.csv")
output_dir <- file.path(root, "output", "Figure4_RF_SHAP_errors")

shap_repetitions <- 200L
random_seed <- 20260926L
descriptive_threshold <- 0.50

# ---- Packages and input checks ----------------------------------------------
packages <- c("ggplot2", "patchwork", "fastshap", "shapviz", "workflows")
missing_packages <- packages[!vapply(
  packages, requireNamespace, logical(1), quietly = TRUE
)]
if (length(missing_packages)) {
  stop(
    "Install required packages: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

missing_files <- c(bundle_path, development_path, external_path)
missing_files <- missing_files[!file.exists(missing_files)]
if (length(missing_files)) {
  stop("Missing input file(s):\n", paste(missing_files, collapse = "\n"),
       call. = FALSE)
}

# ---- Locked model and cohort preparation ------------------------------------
bundle <- readRDS(bundle_path)
required_bundle_fields <- c("fitted_workflow", "selected_model",
                            "selected_predictors")
if (!all(required_bundle_fields %in% names(bundle)) ||
    !identical(as.character(bundle$selected_model), "random_forest") ||
    !inherits(bundle$fitted_workflow, "workflow")) {
  stop("The saved bundle is not the locked random-forest workflow.",
       call. = FALSE)
}

predictors <- as.character(bundle$selected_predictors)
expected_predictors <- c(
  "flank_pain", "stone_attenuation_hu", "proximal_ureter_width",
  "ureteral_wall_thickness", "stone_length", "hydronephrosis_area"
)
if (length(predictors) != 6L || anyDuplicated(predictors) ||
    !setequal(predictors, expected_predictors)) {
  stop("The locked RF does not contain the expected six predictors.",
       call. = FALSE)
}

configuration <- bundle$configuration
if (is.data.frame(configuration) &&
    all(c("setting", "value") %in% names(configuration))) {
  saved_md5 <- as.character(configuration$value[
    configuration$setting == "data_md5"
  ])
  if (length(saved_md5) == 1L && !is.na(saved_md5) &&
      unname(tools::md5sum(development_path)) != saved_md5) {
    stop("The development RDS differs from the data used to lock the RF.",
         call. = FALSE)
  }
}

raw_names <- c(
  "Flank Pain" = "flank_pain",
  "Stone HU" = "stone_attenuation_hu",
  "WUPC" = "proximal_ureter_width",
  "UWT" = "ureteral_wall_thickness",
  "Stone length" = "stone_length",
  "APH" = "hydronephrosis_area",
  "Impacted stone" = "impacted_stone"
)

prepare_cohort <- function(data, label) {
  data <- as.data.frame(data, check.names = FALSE)
  for (from in names(raw_names)) {
    to <- raw_names[[from]]
    if (from %in% names(data) && !to %in% names(data)) {
      names(data)[names(data) == from] <- to
    }
  }
  needed <- c(predictors, "impacted_stone")
  if (!all(needed %in% names(data)) || !nrow(data) ||
      anyNA(data[needed])) {
    stop(label, ": missing columns, empty data, or missing values.",
         call. = FALSE)
  }
  for (name in setdiff(predictors, "flank_pain")) {
    data[[name]] <- suppressWarnings(as.numeric(as.character(data[[name]])))
    if (any(!is.finite(data[[name]]))) {
      stop(label, ": invalid numeric values in ", name, call. = FALSE)
    }
  }
  pain <- trimws(as.character(data$flank_pain))
  if (!all(pain %in% c("No", "Yes"))) {
    stop(label, ": flank_pain must contain only No or Yes.", call. = FALSE)
  }
  data$flank_pain <- factor(pain, levels = c("No", "Yes"))

  outcome <- trimws(as.character(data$impacted_stone))
  outcome[outcome == "Yes"] <- "Impacted"
  outcome[outcome == "No"] <- "Non-impacted"
  if (!all(outcome %in% c("Impacted", "Non-impacted"))) {
    stop(label, ": invalid impacted_stone coding.", call. = FALSE)
  }
  data$impacted_stone <- factor(
    outcome, levels = c("Impacted", "Non-impacted")
  )
  data
}

development <- prepare_cohort(readRDS(development_path), "Development")
external <- prepare_cohort(
  read.csv(external_path, check.names = FALSE, stringsAsFactors = FALSE),
  "External"
)

X_background <- development[, predictors, drop = FALSE]
X_external <- external[, predictors, drop = FALSE]

predict_risk <- function(object, newdata) {
  prediction <- predict(object, new_data = newdata, type = "prob")
  if (!".pred_Impacted" %in% names(prediction)) {
    stop("The locked RF did not return .pred_Impacted.", call. = FALSE)
  }
  as.numeric(prediction$.pred_Impacted)
}

prob_development <- predict_risk(bundle$fitted_workflow, X_background)
prob_external <- predict_risk(bundle$fitted_workflow, X_external)
if (any(!is.finite(prob_development)) || any(prob_development < 0 | prob_development > 1) ||
    any(!is.finite(prob_external)) || any(prob_external < 0 | prob_external > 1)) {
  stop("Predicted probabilities must be finite and within [0, 1].", call. = FALSE)
}

# Reuse a threshold saved with the bundle. Otherwise use the analysis-defined
# 0.50 descriptive cutoff; it is not optimized in this script.
threshold_keys <- c("classification_threshold", "decision_threshold",
                    "probability_threshold", "fixed_threshold")
stored_thresholds <- numeric(0)
if (is.data.frame(configuration) &&
    all(c("setting", "value") %in% names(configuration))) {
  stored_thresholds <- suppressWarnings(as.numeric(as.character(
    configuration$value[configuration$setting %in% threshold_keys]
  )))
}
for (field in threshold_keys) {
  if (field %in% names(bundle)) {
    stored_thresholds <- c(
      stored_thresholds,
      suppressWarnings(as.numeric(bundle[[field]]))
    )
  }
}
stored_thresholds <- unique(stored_thresholds[is.finite(stored_thresholds)])
if (length(stored_thresholds) > 1L ||
    (length(stored_thresholds) &&
     (stored_thresholds <= 0 || stored_thresholds >= 1))) {
  stop("The locked bundle contains conflicting or invalid thresholds.",
       call. = FALSE)
}
threshold <- if (length(stored_thresholds)) {
  stored_thresholds[[1L]]
} else {
  descriptive_threshold
}
threshold_note <- if (length(stored_thresholds)) {
  "locked in bundle"
} else {
  "analysis-defined descriptive threshold"
}

# ---- SHAP calculations -------------------------------------------------------
# All SHAP panels use the complete development cohort as the fixed background.
# SHAP values describe model predictions and are not causal effects.

truth_external <- external$impacted_stone == "Impacted"
predicted_positive <- prob_external >= threshold
error_group <- ifelse(
  truth_external & predicted_positive, "TP",
  ifelse(!truth_external & !predicted_positive, "TN",
         ifelse(!truth_external & predicted_positive, "FP", "FN"))
)
group_levels <- c("TP", "TN", "FP", "FN")
error_group <- factor(error_group, levels = group_levels)
error_counts <- table(error_group)

if (any(error_counts[c("FP", "FN")] == 0L)) {
  stop("Both FP and FN cases are required for the representative waterfalls.",
       call. = FALSE)
}

# Select the case closest to the median predicted probability within each
# error group, as prespecified in the reviewer response.
representative_row <- function(group) {
  rows <- which(error_group == group)
  rows[which.min(abs(prob_external[rows] - stats::median(prob_external[rows])))]
}
fp_row <- representative_row("FP")
fn_row <- representative_row("FN")

X_fp <- X_external[fp_row, , drop = FALSE]
X_fn <- X_external[fn_row, , drop = FALSE]

set.seed(random_seed)
shap_development <- fastshap::explain(
  object = bundle$fitted_workflow,
  X = X_background,
  newdata = X_background,
  pred_wrapper = predict_risk,
  nsim = shap_repetitions,
  adjust = TRUE,
  shap_only = TRUE
)
set.seed(random_seed + 1L)
shap_fp <- fastshap::explain(
  object = bundle$fitted_workflow,
  X = X_background,
  newdata = X_fp,
  pred_wrapper = predict_risk,
  nsim = shap_repetitions,
  adjust = TRUE,
  shap_only = TRUE
)
set.seed(random_seed + 2L)
shap_fn <- fastshap::explain(
  object = bundle$fitted_workflow,
  X = X_background,
  newdata = X_fn,
  pred_wrapper = predict_risk,
  nsim = shap_repetitions,
  adjust = TRUE,
  shap_only = TRUE
)

S_development <- as.matrix(shap_development)
S_fp <- as.matrix(shap_fp)
S_fn <- as.matrix(shap_fn)
baseline_development <- as.numeric(attr(shap_development, "baseline"))
baseline_fp <- as.numeric(attr(shap_fp, "baseline"))
baseline_fn <- as.numeric(attr(shap_fn, "baseline"))

if (length(baseline_development) != 1L ||
    length(baseline_fp) != 1L || length(baseline_fn) != 1L ||
    any(!is.finite(c(baseline_development, baseline_fp, baseline_fn))) ||
    max(abs(c(baseline_fp, baseline_fn) - baseline_development)) > 1e-8 ||
    !identical(colnames(S_development), colnames(X_background)) ||
    !identical(colnames(S_fp), colnames(X_fp)) ||
    !identical(colnames(S_fn), colnames(X_fn)) ||
    nrow(S_development) != nrow(development) ||
    nrow(S_fp) != 1L || nrow(S_fn) != 1L ||
    any(!is.finite(S_development)) || any(!is.finite(S_fp)) ||
    any(!is.finite(S_fn))) {
  stop("SHAP matrices, baselines, or feature order are inconsistent.",
       call. = FALSE)
}

additivity_error_development <- max(
  abs(baseline_development + rowSums(S_development) - prob_development)
)
additivity_error_fp <- abs(
  baseline_fp + sum(S_fp) - prob_external[fp_row]
)
additivity_error_fn <- abs(
  baseline_fn + sum(S_fn) - prob_external[fn_row]
)
if (max(additivity_error_development, additivity_error_fp,
        additivity_error_fn) > 1e-5) {
  stop("SHAP contributions do not reconstruct locked RF probabilities.",
       call. = FALSE)
}

# Units are shown directly in every continuous-variable label.
# UWT and UWA/HNA are CT-derived peri-calculus/obstruction surrogates.
display_name <- c(
  "flank_pain" = "Flank pain (No/Yes)",
  "stone_attenuation_hu" = "Stone attenuation (HU)",
  "proximal_ureter_width" = "Proximal ureter width (mm)",
  "ureteral_wall_thickness" = "Ureteral wall thickness (mm)",
  "stone_length" = "Stone length (mm)",
  "hydronephrosis_area" = "Hydronephrosis area (mm²)"
)

rename_features <- function(shap_matrix, feature_data) {
  colnames(shap_matrix) <- unname(display_name[colnames(shap_matrix)])
  colnames(feature_data) <- unname(display_name[colnames(feature_data)])
  list(shap = shap_matrix, features = feature_data)
}
renamed_development <- rename_features(S_development, X_background)
renamed_fp <- rename_features(S_fp, X_fp)
renamed_fn <- rename_features(S_fn, X_fn)
S_development <- renamed_development$shap
S_fp <- renamed_fp$shap
S_fn <- renamed_fn$shap
X_background <- renamed_development$features
X_fp <- renamed_fp$features
X_fn <- renamed_fn$features
sv_development <- shapviz::shapviz(
  S_development, X = X_background, baseline = baseline_development
)
sv_fp <- shapviz::shapviz(S_fp, X = X_fp, baseline = baseline_fp)
sv_fn <- shapviz::shapviz(S_fn, X = X_fn, baseline = baseline_fn)

# ---- Publication panels ------------------------------------------------------
library(ggplot2)
blue <- "#286D92"
orange <- "#C16A48"
ink <- "#263642"
grey <- "#7E8A92"
red <- "#AE4751"

publication_theme <- theme_classic(base_size = 10, base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 11, colour = ink),
    plot.subtitle = element_text(size = 8.7, colour = "#53636D",
                                 margin = margin(b = 5)),
    plot.title.position = "plot",
    axis.title = element_text(size = 9.2, colour = ink),
    axis.text = element_text(size = 8.3, colour = ink),
    legend.title = element_text(size = 8.5),
    legend.text = element_text(size = 8),
    plot.margin = margin(7, 12, 7, 8)
  )

importance <- sort(colMeans(abs(S_development)), decreasing = TRUE)
importance_data <- data.frame(
  feature = factor(names(importance), levels = rev(names(importance))),
  value = unname(importance)
)

panel_a <- ggplot(importance_data, aes(value, feature)) +
  geom_col(width = 0.68, fill = blue) +
  geom_text(aes(label = sprintf("%.3f", value)), hjust = -0.17,
            size = 3, colour = ink) +
  scale_x_continuous(limits = c(0, max(importance) * 1.24),
                     expand = c(0, 0)) +
  labs(
    title = "A   Global mean absolute SHAP",
    subtitle = sprintf("Development cohort (n = %d); locked RF", nrow(development)),
    x = "Mean absolute SHAP (probability units)", y = NULL
  ) +
  publication_theme

panel_b <- shapviz::sv_importance(
  sv_development,
  kind = "beeswarm",
  max_display = 6L,
  color_bar_title = "Feature value",
  alpha = 0.72,
  size = 1.05
) +
  labs(
    title = "B   SHAP beeswarm",
    subtitle = "Rightward values increase predicted impaction probability",
    x = "SHAP contribution (probability units)", y = NULL
  ) +
  publication_theme +
  theme(legend.position = "right",
        legend.key.height = grid::unit(16, "pt"))

format_case <- function(row, group) {
  sprintf(
    "External case #%d | true: %s | predicted: %s | risk %.3f",
    row,
    if (group == "FP") "non-impacted" else "impacted",
    if (group == "FP") "impacted" else "non-impacted",
    prob_external[row]
  )
}

panel_c <- shapviz::sv_waterfall(
  sv_fp, row_id = 1L, max_display = 6L,
  fill_colors = c(orange, blue), size = 3, annotation_size = 2.8
) +
  labs(
    title = "C   False-positive prediction",
    subtitle = format_case(fp_row, "FP"),
    x = "Predicted impaction probability", y = NULL
  ) +
  publication_theme +
  theme(legend.position = "none")

panel_d <- shapviz::sv_waterfall(
  sv_fn, row_id = 1L, max_display = 6L,
  fill_colors = c(orange, blue), size = 3, annotation_size = 2.8
) +
  labs(
    title = "D   False-negative prediction",
    subtitle = format_case(fn_row, "FN"),
    x = "Predicted impaction probability", y = NULL
  ) +
  publication_theme +
  theme(legend.position = "none")

figure_main <- patchwork::wrap_plots(
  panel_a, panel_b, panel_c, panel_d,
  ncol = 2,
  widths = c(1, 1),
  heights = c(1, 1.03)
)

# Supplementary external probability distributions requested for TP/TN/FP/FN.
external_distribution <- data.frame(
  probability = prob_external,
  group = factor(error_group, levels = group_levels)
)
colour_groups <- c(TP = blue, TN = grey, FP = orange, FN = red)
count_labels <- setNames(
  sprintf("%s (n = %d)", group_levels, as.integer(error_counts)),
  group_levels
)
figure_distribution <- ggplot(
  external_distribution,
  aes(x = probability, y = group, colour = group)
) +
  geom_vline(xintercept = threshold, colour = ink, linetype = "dashed",
             linewidth = 0.55) +
  geom_boxplot(width = 0.44, outlier.shape = NA, fill = "white",
               linewidth = 0.55, orientation = "y") +
  geom_point(position = position_jitter(width = 0, height = 0.11,
                                        seed = random_seed),
             size = 1.25, alpha = 0.65) +
  scale_colour_manual(values = colour_groups, guide = "none", drop = FALSE) +
  scale_y_discrete(limits = group_levels, labels = count_labels, drop = FALSE) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0.015, 0.015))) +
  labs(
    title = "External predicted-probability distributions by outcome group",
    subtitle = sprintf("Locked RF; classification threshold %.2f (%s)",
                        threshold, threshold_note),
    x = "Predicted impaction probability", y = NULL
  ) +
  publication_theme

# ---- Reproducible outputs ----------------------------------------------------
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
main_base <- file.path(output_dir, "Figure4_RF_SHAP_error_analysis")
distribution_base <- file.path(
  output_dir, "Figure4_external_probability_distributions"
)

save_figure <- function(plot, base, width, height) {
  device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
  ggplot2::ggsave(paste0(base, ".pdf"), plot, width = width, height = height,
                  units = "in", device = device, bg = "white")
  ggplot2::ggsave(paste0(base, ".tiff"), plot, width = width, height = height,
                  units = "in", dpi = 600, compression = "lzw", bg = "white")
  ggplot2::ggsave(paste0(base, ".png"), plot, width = width, height = height,
                  units = "in", dpi = 300, bg = "white")
}

save_figure(figure_main, main_base, width = 12.4, height = 9.5)
save_figure(figure_distribution, distribution_base, width = 8.5, height = 5.8)

write.csv(
  data.frame(feature = names(importance),
             mean_absolute_shap = unname(importance)),
  file.path(output_dir, "Figure4_mean_absolute_SHAP.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    external_row = seq_len(nrow(external)),
    actual = as.character(external$impacted_stone),
    predicted = ifelse(predicted_positive, "Impacted", "Non-impacted"),
    predicted_probability = prob_external,
    error_group = as.character(error_group),
    external_fp_or_fn = seq_len(nrow(external)) %in% c(fp_row, fn_row),
    external_case_id = seq_len(nrow(external))
  ),
  file.path(output_dir, "Figure4_external_case_predictions.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(group = group_levels, n = as.integer(error_counts),
             threshold = threshold),
  file.path(output_dir, "Figure4_external_error_counts.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    metric = c(
      "bundle_md5", "development_md5", "external_md5", "development_n",
      "external_n", "external_events", "shap_repetitions", "seed",
      "SHAP_baseline", "development_max_additivity_error",
      "FP_additivity_error", "FN_additivity_error", "FP_representative_row",
      "FN_representative_row", "FP_probability", "FN_probability",
      "classification_threshold", "threshold_provenance"
    ),
    value = as.character(c(
      unname(tools::md5sum(bundle_path)),
      unname(tools::md5sum(development_path)),
      unname(tools::md5sum(external_path)),
      nrow(development), nrow(external), sum(truth_external),
      shap_repetitions, random_seed, baseline_development,
      additivity_error_development, additivity_error_fp, additivity_error_fn,
      fp_row, fn_row, prob_external[fp_row], prob_external[fn_row],
      threshold, threshold_note
    ))
  ),
  file.path(output_dir, "Figure4_audit.csv"),
  row.names = FALSE
)

legend_text <- c(
  "Figure 4. SHAP explanations and external-validation error analysis of the locked random-forest model.",
  sprintf("A: mean absolute SHAP values in the development cohort (n = %d). Values are on the predicted-probability scale and represent contribution magnitude; they do not establish causal importance.", nrow(development)),
  "B: SHAP beeswarm plot. Horizontal position represents the signed contribution to predicted impaction probability; colour represents the within-variable feature value.",
  "C: representative false-positive external case. The true outcome was non-impacted, but the locked RF predicted impacted at the fixed classification threshold.",
  "D: representative false-negative external case. The true outcome was impacted, but the locked RF predicted non-impacted at the fixed classification threshold.",
  sprintf("The FP and FN cases were selected by proximity to the median predicted probability within their respective error groups. Monte Carlo SHAP values used %d repetitions and the complete development cohort as the fixed background distribution.", shap_repetitions),
  "Continuous predictor units: ureteral wall thickness, proximal ureteral width and stone length are reported in mm; hydronephrosis area is reported in mm²; stone attenuation is reported in HU. Flank pain is binary (No/Yes).",
  "SHAP values are model-based post-hoc attributions and should not be interpreted as causal effects or independent adjusted effects."
)
writeLines(legend_text, file.path(output_dir, "Figure4_legend_draft.txt"), useBytes = TRUE)
message("Figure 4 outputs written to: ", output_dir)
