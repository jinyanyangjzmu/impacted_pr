# Figure 4 | Locked RF: SHAP interpretation and external error analysis
# Run: source("C:/Users/Lenovo/Desktop/impacted_pr/Figure4_RF_SHAP_error_analysis.R")
# A/B: all development patients; C/D: external patients. Every SHAP panel
# uses the complete development RDS as its reference. No model fitting.

# ---- Paths and reproducible settings --------------------------------------
root <- "C:/Users/Lenovo/Desktop/impacted_pr"
bundle_path <- file.path(root, "output", "final_selected_model",
                         "final_random_forest_model_bundle.rds")
development_path <- file.path(root, "data", "impacted_clean.rds")
external_path <- file.path(root, "data", "data_wb.csv")
output_dir <- file.path(root, "output", "Figure4_RF_SHAP_errors")
shap_repetitions <- 200L             # Monte Carlo approximations; increase for final analysis
random_seed <- 20260926L
descriptive_threshold <- 0.50        # used only if the bundle has no fixed threshold

packages <- c("ggplot2", "patchwork", "fastshap", "shapviz", "workflows")
missing_packages <- packages[!vapply(packages, requireNamespace,
                                      logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install required packages: ", paste(missing_packages, collapse = ", "))
}
missing_files <- c(bundle_path, development_path, external_path)
missing_files <- missing_files[!file.exists(missing_files)]
if (length(missing_files)) stop("Missing input file(s):\n",
                                paste(missing_files, collapse = "\n"))

# ---- Locked model and cohort checks ---------------------------------------
bundle <- readRDS(bundle_path)
needed <- c("fitted_workflow", "selected_model", "selected_predictors")
if (!all(needed %in% names(bundle)) ||
    !identical(as.character(bundle$selected_model), "random_forest") ||
    !inherits(bundle$fitted_workflow, "workflow")) {
  stop("The saved bundle is not the locked random-forest workflow.")
}
predictors <- as.character(bundle$selected_predictors)
expected <- c("flank_pain", "stone_attenuation_hu", "proximal_ureter_width",
              "ureteral_wall_thickness", "stone_length", "hydronephrosis_area")
if (length(predictors) != 6L || !setequal(predictors, expected)) {
  stop("The locked RF does not have the expected six predictors.")
}

config <- bundle$configuration
if (is.data.frame(config) && all(c("setting", "value") %in% names(config))) {
  saved_md5 <- as.character(config$value[config$setting == "data_md5"])
  if (length(saved_md5) == 1L && !is.na(saved_md5) &&
      unname(tools::md5sum(development_path)) != saved_md5) {
    stop("The development RDS differs from the one used to lock the RF.")
  }
}

raw_names <- c("Flank Pain" = "flank_pain", "Stone HU" = "stone_attenuation_hu",
               "WUPC" = "proximal_ureter_width",
               "UWT" = "ureteral_wall_thickness",
               "Stone length" = "stone_length", "APH" = "hydronephrosis_area",
               "Impacted stone" = "impacted_stone")
prepare <- function(data, label) {
  data <- as.data.frame(data, check.names = FALSE)
  for (from in names(raw_names)) {
    to <- raw_names[[from]]
    if (from %in% names(data) && !to %in% names(data)) {
      names(data)[names(data) == from] <- to
    }
  }
  needed <- c(predictors, "impacted_stone")
  if (!all(needed %in% names(data)) || nrow(data) == 0L ||
      anyNA(data[needed])) stop(label, ": missing or invalid inputs.")
  for (nm in setdiff(predictors, "flank_pain")) {
    data[[nm]] <- suppressWarnings(as.numeric(as.character(data[[nm]])))
    if (any(!is.finite(data[[nm]]))) stop(label, ": invalid ", nm)
  }
  pain <- trimws(as.character(data$flank_pain))
  if (!all(pain %in% c("No", "Yes"))) stop(label, ": invalid flank pain.")
  data$flank_pain <- factor(pain, levels = c("No", "Yes"))
  outcome <- trimws(as.character(data$impacted_stone))
  outcome[outcome == "Yes"] <- "Impacted"
  outcome[outcome == "No"] <- "Non-impacted"
  if (!all(outcome %in% c("Impacted", "Non-impacted"))) {
    stop(label, ": invalid outcome coding.")
  }
  data$impacted_stone <- factor(outcome,
                                levels = c("Impacted", "Non-impacted"))
  data
}

development <- prepare(readRDS(development_path), "Development")
external <- prepare(read.csv(external_path, check.names = FALSE,
                             stringsAsFactors = FALSE), "External")
if (anyDuplicated(predictors)) stop("Duplicated model predictors.")
X_bg <- development[, predictors, drop = FALSE]
X_explain <- external[, predictors, drop = FALSE]

predict_risk <- function(object, newdata) {
  p <- predict(object, new_data = newdata, type = "prob")
  if (!".pred_Impacted" %in% names(p)) {
    stop("Locked RF did not return the probability of 'Impacted'.")
  }
  as.numeric(p$.pred_Impacted)
}
prob <- predict_risk(bundle$fitted_workflow, X_explain)
if (length(prob) != nrow(external) || any(!is.finite(prob)) ||
    any(prob < 0 | prob > 1)) stop("Invalid external RF probabilities.")
development_prob <- predict_risk(bundle$fitted_workflow, X_bg)
if (length(development_prob) != nrow(development) ||
    any(!is.finite(development_prob)) ||
    any(development_prob < 0 | development_prob > 1)) {
  stop("Invalid development RF probabilities.")
}

# If the locked bundle records a probability threshold, reuse it exactly.
# Otherwise use a clearly labelled 0.50 descriptive threshold for panel D.
threshold_keys <- c("classification_threshold", "decision_threshold",
                    "probability_threshold", "fixed_threshold")
stored <- numeric(0)
if (is.data.frame(config) && all(c("setting", "value") %in% names(config))) {
  stored <- suppressWarnings(as.numeric(as.character(
    config$value[config$setting %in% threshold_keys]
  )))
}
for (field in threshold_keys) {
  if (field %in% names(bundle)) {
    stored <- c(stored, suppressWarnings(as.numeric(bundle[[field]])))
  }
}
stored <- unique(stored[is.finite(stored)])
if (length(stored) > 1L || (length(stored) &&
                            (stored <= 0 || stored >= 1))) {
  stop("Conflicting or invalid thresholds in the locked bundle.")
}
threshold <- if (length(stored)) stored else descriptive_threshold
threshold_note <- if (length(stored)) "bundle-locked threshold" else
  "0.50 descriptive threshold; not a bundle-locked cutoff"

# ---- Monte Carlo SHAP on the predicted PROBABILITY scale -----------------
# Both analyses use the same development reference distribution. A/B
# describe the full, refitted RF in its development patients; they are not
# nested out-of-fold predictions or independent model-performance estimates.
# C explains one external patient's prediction. None estimates causal effects.
# Choose the external example by predicted probability, before SHAP analysis.
case_row <- which.min(abs(prob - stats::median(prob)))[1L]
X_case <- X_explain[case_row, , drop = FALSE]

set.seed(random_seed)
shap_development <- fastshap::explain(
  object = bundle$fitted_workflow, X = X_bg,
  newdata = X_bg, pred_wrapper = predict_risk,
  nsim = shap_repetitions, adjust = TRUE, shap_only = TRUE
)
set.seed(random_seed + 1L)
shap_case <- fastshap::explain(
  object = bundle$fitted_workflow, X = X_bg,
  newdata = X_case, pred_wrapper = predict_risk,
  nsim = shap_repetitions, adjust = TRUE, shap_only = TRUE
)
S_dev <- as.matrix(shap_development)
S_case <- as.matrix(shap_case)
baseline_dev <- as.numeric(attr(shap_development, "baseline"))
baseline_case <- as.numeric(attr(shap_case, "baseline"))
if (length(baseline_dev) != 1L || !is.finite(baseline_dev) ||
    length(baseline_case) != 1L || !is.finite(baseline_case) ||
    abs(baseline_dev - baseline_case) > 1e-8 ||
    !identical(colnames(S_dev), colnames(X_bg)) ||
    !identical(colnames(S_case), colnames(X_case)) ||
    nrow(S_dev) != nrow(development) || nrow(S_case) != 1L ||
    any(!is.finite(S_dev)) || any(!is.finite(S_case))) {
  stop("The SHAP matrices, shared baseline, or feature order are inconsistent.")
}
development_error <- max(abs(baseline_dev + rowSums(S_dev) -
                             development_prob))
case_error <- abs(baseline_case + rowSums(S_case) - prob[case_row])
if (max(development_error, case_error) > 1e-5) {
  stop("SHAP contributions do not sum to locked RF probabilities.")
}

display_name <- c(
  "flank_pain" = "Flank pain",
  "stone_attenuation_hu" = "Stone attenuation (HU)",
  "proximal_ureter_width" = "Proximal ureter width",
  "ureteral_wall_thickness" = "Ureteral wall thickness",
  "stone_length" = "Stone length",
  "hydronephrosis_area" = "Hydronephrosis area"
)
colnames(S_dev) <- unname(display_name[colnames(S_dev)])
colnames(S_case) <- unname(display_name[colnames(S_case)])
colnames(X_bg) <- unname(display_name[colnames(X_bg)])
colnames(X_case) <- unname(display_name[colnames(X_case)])
sv_dev <- shapviz::shapviz(S_dev, X = X_bg, baseline = baseline_dev)
sv_case <- shapviz::shapviz(S_case, X = X_case, baseline = baseline_case)

# ---- Four publication panels ----------------------------------------------
library(ggplot2)
blue <- "#286D92"
orange <- "#C16A48"
ink <- "#263642"
grey <- "#8C969C"
theme_pub <- theme_classic(base_size = 10, base_family = "sans") +
  theme(plot.title = element_text(face = "bold", size = 11, colour = ink),
        plot.subtitle = element_text(size = 8.7, colour = "#53636D",
                                     margin = margin(b = 5)),
        plot.title.position = "plot",
        axis.title = element_text(size = 9.1, colour = ink),
        axis.text = element_text(size = 8.3, colour = ink),
        legend.title = element_text(size = 8.5),
        legend.text = element_text(size = 8),
        plot.margin = margin(7, 12, 7, 8))

importance <- sort(colMeans(abs(S_dev)), decreasing = TRUE)
importance_df <- data.frame(
  feature = factor(names(importance), levels = rev(names(importance))),
  value = unname(importance)
)
p_a <- ggplot(importance_df, aes(value, feature)) +
  geom_col(width = 0.68, fill = blue) +
  geom_text(aes(label = sprintf("%.3f", value)), hjust = -0.17,
            size = 3, colour = ink) +
  scale_x_continuous(limits = c(0, max(importance) * 1.24),
                     expand = c(0, 0)) +
  labs(title = "A   Global SHAP importance",
       subtitle = sprintf("Development cohort (n = %d); locked RF", nrow(development)),
       x = "Mean absolute SHAP (probability units)", y = NULL) + theme_pub

p_b <- shapviz::sv_importance(
  sv_dev, kind = "beeswarm", max_display = 6L,
  color_bar_title = "Feature value", alpha = 0.72, size = 1.05
) +
  labs(title = "B   SHAP value and feature value",
       subtitle = "Rightward values increase predicted impaction probability",
       x = "SHAP contribution (probability units)", y = NULL) +
  theme_pub +
  theme(legend.position = "right", legend.key.height = grid::unit(16, "pt"))

p_c <- shapviz::sv_waterfall(
  sv_case, row_id = 1L, max_display = 6L,
  fill_colors = c(orange, blue), size = 3,
  annotation_size = 2.8
) +
  labs(title = "C   One individual prediction",
       subtitle = sprintf("External patient #%d | risk %.3f | baseline %.3f",
                          case_row, prob[case_row], baseline_case),
       x = "Predicted impaction probability", y = NULL) +
  theme_pub + theme(legend.position = "none")

truth <- external$impacted_stone == "Impacted"
predicted_positive <- prob >= threshold
groups <- ifelse(truth & predicted_positive, "TP",
          ifelse(!truth & !predicted_positive, "TN",
          ifelse(!truth & predicted_positive, "FP", "FN")))
group_levels <- c("TP", "TN", "FP", "FN")
counts <- table(factor(groups, levels = group_levels))
distribution <- data.frame(
  risk = prob, group = factor(groups, levels = group_levels)
)
group_labels <- setNames(sprintf("%s  (n=%d)", group_levels,
                                 as.integer(counts)), group_levels)
colors <- c("TP" = blue, "TN" = grey, "FP" = orange, "FN" = "#AE4751")
p_d <- ggplot(distribution, aes(risk, group, colour = group)) +
  geom_vline(xintercept = threshold, colour = ink,
             linetype = "dashed", linewidth = 0.55) +
  geom_boxplot(width = 0.42, outlier.shape = NA, fill = "white",
               linewidth = 0.55, orientation = "y") +
  geom_point(position = position_jitter(width = 0, height = 0.11,
                                        seed = random_seed),
             size = 1.25, alpha = 0.63) +
  scale_colour_manual(values = colors, guide = "none", drop = FALSE) +
  scale_y_discrete(limits = group_levels, labels = group_labels, drop = FALSE) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0.015, 0.015))) +
  labs(title = "D   External classification outcomes",
       subtitle = sprintf("Cutoff %.2f | %s", threshold,
                          if (length(stored)) "locked in bundle" else "descriptive"),
       x = "Locked RF predicted impaction probability", y = NULL) +
  theme_pub

figure <- patchwork::wrap_plots(
  patchwork::wrap_plots(p_a, p_b, ncol = 2, widths = c(0.96, 1.04)),
  patchwork::wrap_plots(p_c, p_d, ncol = 2, widths = c(1.12, 0.88)),
  ncol = 1, heights = c(1, 1.03)
)

# ---- Reproducible outputs: figure, numerical summaries, and legend --------
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
base_file <- file.path(output_dir, "Figure4_RF_SHAP_error_analysis")
pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
ggsave(paste0(base_file, ".pdf"), figure, width = 12.4, height = 9.5,
       units = "in", device = pdf_device, bg = "white")
ggsave(paste0(base_file, ".tiff"), figure, width = 12.4, height = 9.5,
       units = "in", dpi = 600, compression = "lzw", bg = "white")
ggsave(paste0(base_file, ".png"), figure, width = 12.4, height = 9.5,
       units = "in", dpi = 300, bg = "white")

write.csv(data.frame(feature = names(importance), mean_absolute_shap =
                       unname(importance)),
          file.path(output_dir, "Figure4_mean_absolute_SHAP.csv"),
          row.names = FALSE)
write.csv(data.frame(group = group_levels, n = as.integer(counts),
                     threshold = threshold),
          file.path(output_dir, "Figure4_external_error_counts.csv"),
          row.names = FALSE)
write.csv(data.frame(metric = c("bundle_md5", "development_md5", "external_md5",
                                 "development_n", "external_n", "external_events",
                                 "shap_repetitions", "seed", "SHAP_baseline",
                                 "development_max_additivity_error",
                                 "external_case_additivity_error",
                                 "representative_row", "representative_probability",
                                 "classification_threshold", "threshold_provenance"),
                     value = as.character(c(
                       unname(tools::md5sum(bundle_path)),
                       unname(tools::md5sum(development_path)),
                       unname(tools::md5sum(external_path)), nrow(development),
                       nrow(external), sum(truth), shap_repetitions, random_seed,
                       baseline_dev, development_error, case_error, case_row,
                       prob[case_row], threshold, threshold_note))),
          file.path(output_dir, "Figure4_audit.csv"), row.names = FALSE)
writeLines(c(
  "Figure 4. SHAP explanations and external-validation error analysis of the locked random-forest model.",
  "A and B explain predictions of the locked final RF in all development patients.",
  "These are descriptive, in-sample explanations of the refitted final model,",
  "not nested out-of-fold predictions or independent performance estimates.",
  "C explains one external-validation patient; all SHAP panels use the complete",
  "development cohort as the fixed reference population. Monte Carlo SHAP values",
  sprintf("were calculated with %d repetitions per feature and local additivity adjustment.",
          shap_repetitions),
  "All SHAP values refer to the predicted probability of an impacted stone.",
  "A: mean absolute SHAP is a global summary of contribution magnitude across",
  "development patients; it does not indicate effect direction.",
  "B: SHAP beeswarm; position represents signed contribution to predicted risk,",
  "color represents within-variable feature value (flank pain: No to Yes).",
  "C: waterfall for the external patient whose predicted probability is closest",
  "to the external cohort median; baseline is mean locked-RF prediction in the",
  "development reference population. This panel explains one prediction only.",
  sprintf("D: predicted probabilities by TP/TN/FP/FN at threshold %.2f (%s).",
          threshold, threshold_note),
  "TP = true positive; TN = true negative; FP = false positive; FN = false negative.",
  "SHAP attributes model predictions under the chosen reference distribution.",
  "It does not estimate causal or independent predictor effects."
), file.path(output_dir, "Figure4_legend_draft.txt"), useBytes = TRUE)
message("Figure 4 written to ", output_dir)
