# Table 2: nine-algorithm comparison from the completed fully nested CV.
# Run: source("C:/Users/Lenovo/Desktop/impacted_pr/Table2_nested_algorithm_comparison_revised.R")
# Requires only flextable and officer; no model refitting or new thresholds.

# ---- Paths ---------------------------------------------------------------
root <- "C:/Users/Lenovo/Desktop/impacted_pr"
results_file <- file.path(root, "output", "fully_nested_lasso_9models",
                          "fully_nested_lasso_9models_results.rds")
bundle_file <- file.path(root, "output", "final_selected_model",
                         "final_random_forest_model_bundle.rds")
output_dir <- file.path(root, "output", "Table2_nested_algorithm_comparison_revised")

packages <- c("flextable", "officer")
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1),
                                      quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install required packages: install.packages(c(",
       paste(sprintf('"%s"', missing_packages), collapse = ", "), "))")
}
if (!file.exists(results_file) || !file.exists(bundle_file)) {
  stop("Missing nested results RDS or final RF bundle. Check the paths above.")
}

res <- readRDS(results_file)
bundle <- readRDS(bundle_file)
required <- c("all_model_summary", "all_model_fold_metrics",
              "algorithm_selection_frequency", "selected_model", "settings")
if (!all(required %in% names(res))) {
  stop("Nested results missing: ", paste(setdiff(required, names(res)), collapse = ", "))
}
if (!identical(as.character(res$selected_model), "random_forest") ||
    !identical(as.character(bundle$selected_model), "random_forest")) {
  stop("The nested analysis and final bundle must identify the locked RF.")
}

# Prevent accidental mixing of results from another model-development run.
config <- bundle$configuration
if (!is.data.frame(config) || !all(c("setting", "value") %in% names(config))) {
  stop("The model bundle lacks the configuration used to verify provenance.")
}
recorded_md5 <- as.character(config$value[config$setting == "nested_results_md5"])
if (length(recorded_md5) != 1L ||
    unname(tools::md5sum(results_file)) != recorded_md5) {
  stop("The nested results RDS differs from the file used to lock the RF.")
}

summ <- as.data.frame(res$all_model_summary)
fold <- as.data.frame(res$all_model_fold_metrics)
freq <- as.data.frame(res$algorithm_selection_frequency)
models <- c("logistic", "elastic_net", "decision_tree", "random_forest",
            "svm_rbf", "knn", "lightgbm", "mlp", "xgboost")
need_summary <- c("model", "model_label", "outer_auc_mean", "outer_auc_sd",
                  "outer_brier_mean", "completed_outer_folds", "expected_outer_folds",
                  "failed_outer_folds", "n_patients", "complete_oof_coverage")
need_fold <- c("model", "outer_id", "roc_auc", "brier")
need_freq <- c("model", "selected_outer_folds", "selection_frequency")
if (!all(need_summary %in% names(summ)) ||
    !all(need_fold %in% names(fold)) ||
    !all(need_freq %in% names(freq))) {
  stop("The nested results do not include the outer-fold and selection audit fields.")
}
if (nrow(summ) != 9L || nrow(freq) != 9L || nrow(fold) != 225L) {
  stop("Unexpected number of nine-model summary or fold records.")
}
if (anyDuplicated(summ$model) || anyDuplicated(freq$model) ||
    !setequal(summ$model, models) || !setequal(freq$model, models) ||
    !setequal(unique(fold$model), models) ||
    anyDuplicated(paste(fold$model, fold$outer_id, sep = "|"))) {
  stop("The nine algorithms or their fold identifiers are missing/duplicated.")
}
if (!all(is.finite(fold$roc_auc)) || !all(is.finite(fold$brier)) ||
    any(!is.finite(as.matrix(summ[c("outer_auc_mean", "outer_auc_sd",
                                     "outer_brier_mean")]))) ||
    any(fold$roc_auc < 0 | fold$roc_auc > 1 | fold$brier < 0 | fold$brier > 1) ||
    any(summ$expected_outer_folds != 25L) ||
    any(summ$completed_outer_folds != 25L) ||
    any(summ$failed_outer_folds != 0L) ||
    any(summ$n_patients != 545L) ||
    any(is.na(summ$complete_oof_coverage) | !summ$complete_oof_coverage)) {
  stop("The completed 25-fold, 545-patient nine-model validation is required.")
}
reference_folds <- sort(unique(as.character(fold$outer_id[fold$model == models[1L]])))
if (length(reference_folds) != 25L ||
    any(vapply(models, function(m) !setequal(
      as.character(fold$outer_id[fold$model == m]), reference_folds), logical(1L)))) {
  stop("Models do not share the same 25 outer-fold identifiers.")
}

# Check saved values against the original fold metrics; never substitute
# pooled patient-level OOF Brier for mean outer-assessment Brier.
for (m in models) {
  f <- fold[fold$model == m, , drop = FALSE]
  s <- summ[summ$model == m, , drop = FALSE]
  if (nrow(f) != 25L || nrow(s) != 1L ||
      any(abs(c(mean(f$roc_auc) - s$outer_auc_mean,
                stats::sd(f$roc_auc) - s$outer_auc_sd,
                mean(f$brier) - s$outer_brier_mean)) > 1e-8)) {
    stop("Saved outer-fold summary differs from raw fold metrics: ", m)
  }
}
if (anyNA(freq$selected_outer_folds) || anyNA(freq$selection_frequency) ||
    sum(freq$selected_outer_folds) != 25L ||
    any(freq$selected_outer_folds < 0L) ||
    any(abs(freq$selected_outer_folds / 25 - freq$selection_frequency) > 1e-10)) {
  stop("Inner algorithm-choice frequencies do not sum to 25 folds.")
}

# ---- Concise manuscript table --------------------------------------------
order <- order(-summ$outer_auc_mean, match(summ$model, models))
summ <- summ[order, , drop = FALSE]
freq <- freq[match(summ$model, freq$model), , drop = FALSE]
if (anyNA(freq$model) || !identical(as.character(summ$model),
                                    as.character(freq$model))) {
  stop("Could not align algorithm selection counts to model summaries.")
}
raw <- data.frame(
  Model = as.character(summ$model_label),
  outer_auc_mean = as.numeric(summ$outer_auc_mean),
  outer_auc_sd = as.numeric(summ$outer_auc_sd),
  outer_brier_mean = as.numeric(summ$outer_brier_mean),
  selected_outer_folds = as.integer(freq$selected_outer_folds),
  selection_frequency = as.numeric(freq$selection_frequency),
  completed_outer_folds = as.integer(summ$completed_outer_folds),
  expected_outer_folds = as.integer(summ$expected_outer_folds),
  check.names = FALSE
)
tab <- data.frame(
  Model = raw$Model,
  `Outer AUC\nmean (SD)` = sprintf("%.3f (%.3f)", raw$outer_auc_mean,
                                  raw$outer_auc_sd),
  `Outer Brier\nmean` = sprintf("%.3f", raw$outer_brier_mean),
  `Selected\nfolds, n (%)` = sprintf("%d/25 (%.0f%%)",
                                     raw$selected_outer_folds,
                                     100 * raw$selection_frequency),
  `Completed\nfolds` = sprintf("%d/%d", raw$completed_outer_folds,
                                raw$expected_outer_folds),
  check.names = FALSE
)
if (nrow(tab) != 9L || sum(summ$model == "random_forest") != 1L) {
  stop("The final nine-model table did not include the locked RF.")
}

# A quiet journal table: horizontal rules, no vertical grid, one selected-row
# cue. Caption and explanatory note are separate Word paragraphs.
ft <- flextable::flextable(tab)
ft <- flextable::theme_booktabs(ft)
ft <- flextable::set_table_properties(ft, layout = "fixed",
                                     opts_word = list(split = FALSE))
ft <- flextable::width(ft, j = seq_len(5L),
                       width = c(2.05, 1.16, 1.07, 1.18, 0.88))
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 9, part = "all")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::valign(ft, valign = "center", part = "all")
ft <- flextable::align(ft, j = 1L, align = "left", part = "all")
ft <- flextable::align(ft, j = 2:5, align = "center", part = "all")
ft <- flextable::padding(ft, padding.top = 5, padding.bottom = 5,
                         padding.left = 4, padding.right = 4, part = "all")
rf_row <- which(summ$model == "random_forest")
ft <- flextable::bold(ft, i = rf_row, part = "body")
ft <- flextable::bg(ft, i = rf_row, bg = "#EEF2F5", part = "body")

title <- "Table 2. Comparison of nine algorithms in fully nested cross-validation"
notes <- paste(
  "Data: 545 development patients; 25 outer assessments per algorithm.",
  "AUC and Brier are the mean of outer-assessment metrics; AUC SD describes",
  "variation across overlapping repeated folds and is not a confidence interval.",
  "Selections use inner-training results only: models within 0.01 of the best",
  "AUC and 0.005 of the best Brier were assessed using calibration error",
  "(|intercept| + |slope - 1|, within 0.10 of the best); the prespecified",
  "complexity order resolved practical ties.",
  sprintf("Random forest (shaded) was selected most often (%d/25); the outer AUC ranking",
          freq$selected_outer_folds[rf_row]),
  "did not determine the final model. Completed folds indicate valid outer",
  "assessments. AUC, area under the receiver operating characteristic curve."
)

doc <- officer::read_docx()
doc <- officer::body_add_fpar(doc, officer::fpar(
  officer::ftext(title, prop = officer::fp_text(font.family = "Arial",
                  font.size = 10, bold = TRUE, color = "#222222")),
  fp_p = officer::fp_par(keep_with_next = TRUE)
))
doc <- flextable::body_add_flextable(doc, ft)
doc <- officer::body_add_fpar(doc, officer::fpar(
  officer::ftext(notes, prop = officer::fp_text(font.family = "Arial",
                  font.size = 8.5, color = "#333333"))
))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
print(doc, target = file.path(output_dir, "Table2_nested_algorithm_comparison_revised.docx"))
utils::write.csv(raw, file.path(output_dir, "Table2_nested_algorithm_comparison_raw.csv"),
                 row.names = FALSE)
utils::write.csv(tab, file.path(output_dir, "Table2_nested_algorithm_comparison_formatted.csv"),
                 row.names = FALSE)
message("Table 2 and audit CSVs saved in: ",
        normalizePath(output_dir, winslash = "/"))
