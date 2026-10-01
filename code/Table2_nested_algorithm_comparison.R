# Table 2 | Nine-algorithm fully nested CV comparison
# Reads the existing nested-CV results and the locked RF bundle. No refitting.
# Outputs editable Word (.docx), vector PDF, 600-dpi TIFF, PNG and CSVs.
# Usage: source("C:/Users/Lenovo/Desktop/impacted_pr/Table2_nested_algorithm_comparison.R")
# Or: Rscript Table2_nested_algorithm_comparison.R [results.rds] [bundle.rds] [output_dir]

root <- "C:/Users/Lenovo/Desktop/impacted_pr"
args <- commandArgs(trailingOnly = TRUE)
results_path <- if (length(args) >= 1L) args[[1]] else
  file.path(root, "output", "fully_nested_lasso_9models",
            "fully_nested_lasso_9models_results.rds")
bundle_path <- if (length(args) >= 2L) args[[2]] else
  file.path(root, "output", "final_selected_model",
            "final_random_forest_model_bundle.rds")
output_dir <- if (length(args) >= 3L) args[[3]] else
  file.path(root, "output", "Table2_nested_algorithm_comparison")
bootstrap_reps <- 2000L
bootstrap_seed <- 20260926L

word_packages <- c("flextable", "officer")
missing_word_packages <- word_packages[
  !vapply(word_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_word_packages)) {
  stop("Install the Word-export packages first: install.packages(c(",
       paste(sprintf('"%s"', missing_word_packages), collapse = ", "), "))")
}

if (!file.exists(results_path) || !file.exists(bundle_path)) {
  stop("The nested results RDS or locked RF bundle was not found.\n",
       results_path, "\n", bundle_path)
}
res <- readRDS(results_path)
bundle <- readRDS(bundle_path)
required <- c("all_model_summary", "all_model_patient_oof",
              "all_model_fold_metrics", "algorithm_selection_frequency",
              "selected_model", "settings")
if (!all(required %in% names(res))) {
  stop("Nested results are missing: ",
       paste(setdiff(required, names(res)), collapse = ", "))
}
if (!identical(as.character(res$selected_model), "random_forest") ||
    !identical(as.character(bundle$selected_model), "random_forest")) {
  stop("The nested selection and locked bundle must both identify random_forest.")
}
if (is.data.frame(bundle$configuration) &&
    all(c("setting", "value") %in% names(bundle$configuration))) {
  saved_md5 <- as.character(bundle$configuration$value[
    bundle$configuration$setting == "nested_results_md5"])
  if (length(saved_md5) == 1L && !is.na(saved_md5) &&
      unname(tools::md5sum(results_path)) != saved_md5) {
    stop("The nested results RDS differs from the results used to lock the RF.")
  }
}

summ <- as.data.frame(res$all_model_summary)
fold <- as.data.frame(res$all_model_fold_metrics)
oof <- as.data.frame(res$all_model_patient_oof)
freq <- as.data.frame(res$algorithm_selection_frequency)
columns <- c("model", "model_label", "outer_auc_mean", "outer_auc_sd",
             "pooled_oof_auc", "pooled_oof_brier", "calibration_intercept",
             "calibration_slope", "completed_outer_folds",
             "expected_outer_folds", "failed_outer_folds", "n_patients",
             "complete_oof_coverage")
if (!all(columns %in% names(summ)) ||
    !all(c("model", "outer_id", "roc_auc") %in% names(fold)) ||
    !all(c("model", "patient_id", "truth", "probability",
           "n_predictions") %in% names(oof)) ||
    !all(c("model", "selected_outer_folds", "selection_frequency") %in%
         names(freq))) {
  stop("Nested results do not have the columns written by the supplied CV script.")
}
models <- c("logistic", "elastic_net", "decision_tree", "random_forest",
            "svm_rbf", "knn", "lightgbm", "mlp", "xgboost")
if (nrow(summ) != 9L || anyDuplicated(summ$model) ||
    !setequal(summ$model, models) || nrow(freq) != 9L ||
    anyDuplicated(freq$model) || !setequal(freq$model, models)) {
  stop("Exactly nine distinct algorithms are required in both summary tables.")
}
outer_repeats <- as.integer(as.character(res$settings$value[
  res$settings$setting == "outer_repeats"]))
if (length(outer_repeats) != 1L || !is.finite(outer_repeats) ||
    outer_repeats < 1L) stop("Could not identify the number of outer repeats.")
expected_folds <- unique(as.integer(summ$expected_outer_folds))
expected_patients <- unique(as.integer(summ$n_patients))
if (length(expected_folds) != 1L || expected_folds != 25L ||
    length(expected_patients) != 1L || expected_patients != 545L ||
    any(summ$completed_outer_folds != expected_folds) ||
    any(summ$failed_outer_folds != 0L) ||
    any(is.na(summ$complete_oof_coverage) |
        !summ$complete_oof_coverage) ||
    nrow(fold) != 9L * expected_folds ||
    any(!is.finite(fold$roc_auc))) {
  stop("All nine models must have complete 25-fold / 545-patient coverage.")
}
if (nrow(oof) != 9L * expected_patients ||
    anyDuplicated(paste(oof$model, oof$patient_id, sep = "|")) ||
    any(oof$n_predictions != outer_repeats) ||
    any(!is.finite(oof$probability)) ||
    any(oof$probability < 0 | oof$probability > 1) ||
    !all(as.character(oof$truth) %in% c("Impacted", "Non-impacted"))) {
  stop("Patient-level out-of-fold probabilities are incomplete or invalid.")
}
ids_reference <- as.character(oof$patient_id[oof$model == models[[1L]]])
if (length(ids_reference) != expected_patients ||
    any(vapply(models, function(m) !setequal(ids_reference,
        as.character(oof$patient_id[oof$model == m])), logical(1)))) {
  stop("Patient IDs differ across the nine model OOF datasets.")
}
if (sum(freq$selected_outer_folds) != expected_folds ||
    any(freq$selected_outer_folds < 0 | freq$selected_outer_folds > expected_folds) ||
    any(abs(freq$selection_frequency -
            freq$selected_outer_folds / expected_folds) > 1e-9)) {
  stop("Algorithm-choice counts are inconsistent with the outer folds.")
}

# The CI is for patient-level pooled OOF AUC. It conditions on the stored
# predictions and does not include uncertainty from re-running model fitting.
# Repeated outer folds are correlated; do not apply a t interval to 25 fold AUCs.
auc_rank <- function(y, p) {
  n1 <- sum(y == 1L)
  n0 <- sum(y == 0L)
  if (!n1 || !n0) stop("AUC requires both outcome classes.")
  (sum(rank(p, ties.method = "average")[y == 1L]) -
     n1 * (n1 + 1) / 2) / (n1 * n0)
}
ci_one_model <- function(model, model_index) {
  d <- oof[oof$model == model, , drop = FALSE]
  y <- as.integer(as.character(d$truth) == "Impacted")
  p <- as.numeric(d$probability)
  recorded <- as.numeric(summ$pooled_oof_auc[summ$model == model])
  if (abs(auc_rank(y, p) - recorded) > 1e-6) {
    stop("Recomputed OOF AUC disagrees with saved summary for: ", model)
  }
  pos <- which(y == 1L)
  neg <- which(y == 0L)
  set.seed(bootstrap_seed + model_index)
  replicate_auc <- replicate(bootstrap_reps, {
    index <- c(sample(pos, length(pos), replace = TRUE),
               sample(neg, length(neg), replace = TRUE))
    auc_rank(y[index], p[index])
  })
  as.numeric(quantile(replicate_auc, probs = c(0.025, 0.975),
                      names = FALSE, type = 7))
}
ci <- t(vapply(seq_along(models), function(i) ci_one_model(models[i], i),
               numeric(2)))
ci <- data.frame(model = models, auc_ci_low = ci[, 1], auc_ci_high = ci[, 2])

data <- merge(summ, freq[c("model", "selected_outer_folds",
                            "selection_frequency")], by = "model", sort = FALSE)
data <- merge(data, ci, by = "model", sort = FALSE)
data <- data[order(-data$outer_auc_mean, data$model), , drop = FALSE]
row.names(data) <- NULL
if (any(!is.finite(data$outer_auc_mean)) ||
    any(!is.finite(data$outer_auc_sd)) ||
    any(!is.finite(data$pooled_oof_brier)) ||
    any(!is.finite(data$auc_ci_low)) ||
    any(!is.finite(data$auc_ci_high))) {
  stop("Non-finite numerical results cannot be published in Table 2.")
}
fmt <- function(x) ifelse(is.finite(x), sprintf("%.3f", x), "NA")
table_data <- data.frame(
  Model = paste0(data$model_label,
                 ifelse(data$model == "random_forest", " *", "")),
  Outer_AUC = sprintf("%.3f (%.3f)", data$outer_auc_mean, data$outer_auc_sd),
  OOF_AUC = sprintf("%.3f (%.3f-%.3f)", data$pooled_oof_auc,
                    data$auc_ci_low, data$auc_ci_high),
  Brier = fmt(data$pooled_oof_brier),
  Intercept = fmt(data$calibration_intercept),
  Slope = fmt(data$calibration_slope),
  Selection = sprintf("%d/25 (%.0f%%)", data$selected_outer_folds,
                      100 * data$selection_frequency),
  check.names = FALSE, stringsAsFactors = FALSE
)
names(table_data) <- c("Model", "Outer-fold AUC\nmean (SD)",
                       "Pooled OOF AUC\n(95% CI)", "Pooled OOF\nBrier",
                       "Calibration\nintercept", "Calibration\nslope",
                       "Selected folds\nn/25 (%)")
table_notes <- c(
  "Outer-fold AUC: mean and SD across 25 outer assessments; fold results overlap across repeats, so SD is descriptive.",
  sprintf("Pooled OOF AUC: patient probabilities averaged across repeats; 95%% CI from %s stratified patient bootstrap resamples.",
          format(bootstrap_reps, big.mark = ",")),
  "The CI conditions on saved OOF predictions and does not account for uncertainty from refitting the nested procedure.",
  "Pooled OOF Brier and calibration intercept/slope use the same averaged probabilities; ideal: 0, 0 and 1; NA: not estimable.",
  "Selected folds: inner-training-only algorithm choices; * locked final RF selected by frequency, not outer assessment AUC."
)

# grid draws text at fixed column widths; PDF remains editable vector text.
# No browser, screenshot package, or Arial font is needed to export the table.
draw_table <- function() {
  grid::grid.newpage()
  navy <- "#223847"; blue <- "#226786"; muted <- "#56636B"
  faint <- "#F3F7F9"; selected_fill <- "#E7F1F6"; rule <- "#C7D3D9"
  grid::grid.rect(gp = grid::gpar(fill = "white", col = NA))
  grid::grid.text("Table 2. Fully nested comparison of nine algorithms",
                  x = 0.046, y = 0.961, default.units = "npc", just = "left",
                  gp = grid::gpar(fontsize = 15, fontface = "bold", col = navy))
  grid::grid.text(
    sprintf("Development cohort (n = %d)  |  %d outer assessments per algorithm  |  Locked final model: random forest",
            expected_patients, expected_folds),
    x = 0.046, y = 0.916, default.units = "npc", just = "left",
    gp = grid::gpar(fontsize = 9.6, col = muted))
  grid::grid.lines(x = c(0.046, 0.954), y = c(0.888, 0.888),
                   gp = grid::gpar(col = blue, lwd = 1.35))

  widths <- c(0.19, 0.17, 0.21, 0.10, 0.11, 0.09, 0.13)
  lefts <- 0.046 + c(0, head(cumsum(widths), -1)) * 0.908
  centers <- lefts + widths * 0.908 / 2
  labels <- names(table_data)
  for (j in seq_along(labels)) {
    grid::grid.text(labels[[j]], x = if (j == 1L) lefts[[j]] else centers[[j]],
                    y = 0.839, default.units = "npc",
                    just = if (j == 1L) "left" else "center",
                    gp = grid::gpar(fontsize = 9.1, fontface = "bold", col = navy,
                                    lineheight = 1.12))
  }
  grid::grid.lines(x = c(0.046, 0.954), y = c(0.790, 0.790),
                   gp = grid::gpar(col = rule, lwd = 0.8))
  for (i in seq_len(nrow(table_data))) {
    yy <- 0.757 - (i - 1L) * 0.0585
    selected <- data$model[[i]] == "random_forest"
    if (selected || i %% 2L == 0L) {
      grid::grid.rect(x = 0.5, y = yy, width = 0.908, height = 0.058,
                      gp = grid::gpar(fill = if (selected) selected_fill else faint,
                                      col = NA))
    }
    for (j in seq_along(labels)) {
      grid::grid.text(as.character(table_data[i, j]),
                      x = if (j == 1L) lefts[[j]] + 0.004 else centers[[j]],
                      y = yy, default.units = "npc",
                      just = if (j == 1L) "left" else "center",
                      gp = grid::gpar(fontsize = 9.2,
                                      fontface = if (selected) "bold" else "plain",
                                      col = if (selected) blue else navy))
    }
  }
  grid::grid.lines(x = c(0.046, 0.954), y = c(0.256, 0.256),
                   gp = grid::gpar(col = rule, lwd = 0.8))
  for (k in seq_along(table_notes)) {
    grid::grid.text(table_notes[[k]], x = 0.046,
                    y = 0.222 - (k - 1L) * 0.041,
                    default.units = "npc", just = "left",
                    gp = grid::gpar(fontsize = 8.15, col = muted))
  }
}

# A real Word table: every cell, title, and footnote can be edited in Word.
make_editable_word_table <- function() {
  word_data <- table_data
  names(word_data) <- c("Model", "Outer_AUC", "Pooled_AUC", "Brier",
                        "Intercept", "Slope", "Selection")
  ft <- flextable::flextable(word_data)
  ft <- flextable::set_header_labels(ft, values = unname(names(table_data)))
  ft <- flextable::add_header_lines(
    ft, values = sprintf("Development cohort (n = %d) | %d outer assessments per algorithm | Locked final model: random forest",
                         expected_patients, expected_folds))
  ft <- flextable::add_header_lines(
    ft, values = "Table 2. Fully nested comparison of nine algorithms")
  ft <- flextable::add_footer_lines(ft, values = table_notes)
  ft <- flextable::border_remove(ft)
  ft <- flextable::width(ft, j = names(word_data),
                         width = c(1.74, 1.44, 1.83, 0.91, 1.12, 0.88, 1.33))
  ft <- flextable::font(ft, fontname = "Arial", part = "all")
  ft <- flextable::fontsize(ft, size = 9, part = "body")
  ft <- flextable::fontsize(ft, size = 8.8, part = "header")
  ft <- flextable::fontsize(ft, size = 8, part = "footer")
  ft <- flextable::fontsize(ft, i = 1, size = 13, part = "header")
  ft <- flextable::fontsize(ft, i = 2, size = 9, part = "header")
  ft <- flextable::bold(ft, part = "header")
  ft <- flextable::bold(ft, i = which(data$model == "random_forest"),
                         part = "body")
  ft <- flextable::bg(ft, bg = "#E8F0F4", part = "header")
  ft <- flextable::bg(ft, i = seq(2L, nrow(word_data), by = 2L),
                       bg = "#F3F7F9", part = "body")
  ft <- flextable::bg(ft, i = which(data$model == "random_forest"),
                       bg = "#DDEBF2", part = "body")
  ft <- flextable::color(ft, i = which(data$model == "random_forest"),
                          color = "#226786", part = "body")
  ft <- flextable::align(ft, align = "center", part = "all")
  ft <- flextable::align(ft, j = "Model", align = "left", part = "body")
  ft <- flextable::align(ft, align = "left", part = "footer")
  ft <- flextable::valign(ft, valign = "center", part = "all")
  ft <- flextable::padding(ft, padding = 3, part = "header")
  ft <- flextable::padding(ft, padding = 4, part = "body")
  ft <- flextable::padding(ft, padding = 2, part = "footer")
  ft <- flextable::hline_bottom(
    ft, part = "header",
    border = officer::fp_border(color = "#AEBEC8", width = 1))
  ft <- flextable::hline_top(
    ft, part = "footer",
    border = officer::fp_border(color = "#AEBEC8", width = 1))
  flextable::set_table_properties(ft, layout = "fixed", width = 1)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(output_dir, "Table2_nested_algorithm_comparison")
word_table <- make_editable_word_table()
word_section <- officer::prop_section(
  page_size = officer::page_size(orient = "landscape"),
  page_margins = officer::page_mar(top = 0.56, bottom = 0.56,
                                   left = 0.60, right = 0.60))
flextable::save_as_docx(word_table, path = paste0(stem, ".docx"),
                        pr_section = word_section, align = "center")
write.csv(table_data, paste0(stem, "_formatted.csv"), row.names = FALSE,
          fileEncoding = "UTF-8")
write.csv(data[c("model", "model_label", "outer_auc_mean", "outer_auc_sd",
                 "pooled_oof_auc", "auc_ci_low", "auc_ci_high",
                 "pooled_oof_brier", "calibration_intercept", "calibration_slope",
                 "selected_outer_folds", "selection_frequency",
                 "completed_outer_folds", "n_patients")],
          paste0(stem, "_numeric_audit.csv"), row.names = FALSE)
pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
export_table <- function(path, open_device) {
  open_device(path)
  on.exit(grDevices::dev.off(), add = TRUE)
  draw_table()
}
export_table(paste0(stem, ".pdf"), function(path)
  pdf_device(path, width = 12.8, height = 6.8, family = "sans", bg = "white"))
export_table(paste0(stem, ".tiff"), function(path)
  grDevices::tiff(path, width = 12.8, height = 6.8, units = "in",
                 res = 600, compression = "lzw", bg = "white"))
export_table(paste0(stem, ".png"), function(path)
  grDevices::png(path, width = 12.8, height = 6.8, units = "in",
                res = 300, bg = "white"))
message("Saved Table 2 (editable DOCX, PDF/TIFF/PNG, formatted CSV, numeric audit): ", stem)
