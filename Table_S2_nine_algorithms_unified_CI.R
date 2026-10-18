# ============================================================
# Table S2: nine algorithms, patient-level nested OOF predictions
# Unified with Table2_nested_algorithm_comparison_unified_CI.R
# and Table3_final_RF_performance.R.
#
# AUC: DeLong 95% CI.
# Classification proportions: exact binomial 95% CI.
# F1/MCC/Youden/Brier/calibration: 2,000 ordinary patient
# bootstrap resamples, percentile intervals (no stratification).
# No model refitting or threshold optimization.
#
# Run with the same R RNG settings used for Tables 2 and 3.
# The full nested-results RDS, not the final RF bundle, is needed.
# install.packages(c("pROC", "officer", "flextable"))
# ============================================================

required_packages <- c("pROC", "officer", "flextable")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Install packages: ", paste(missing_packages, collapse = ", "))
}

# ---- 1. Paths and settings -----------------------------------------------
root <- "C:/Users/Lenovo/Desktop/impacted_pr"
nested_file <- file.path(
  root, "output", "fully_nested_lasso_9models",
  "fully_nested_lasso_9models_results.rds"
)
output_dir <- file.path(root, "output", "Table_S2_publication_unified_CI")

# Optional audit CSVs. Missing files are reported; checks are not fabricated.
table2_audit_file <- file.path(
  root, "output", "Table2_nested_algorithm_comparison_unified_CI",
  "Table2_nested_algorithm_comparison_unified_CI_raw.csv"
)
table3_audit_file <- file.path(
  root, "output", "Table3_final_RF_performance",
  "Table3_final_RF_performance_numeric_audit.csv"
)
positive <- "Impacted"
negative <- "Non-impacted"
threshold <- 0.50
bootstrap_reps <- 2000L
seed <- 20260926L
conf_level <- 0.95
models <- c(
  "logistic", "elastic_net", "decision_tree", "random_forest",
  "svm_rbf", "knn", "lightgbm", "mlp", "xgboost"
)
model_labels <- c(
  logistic = "Logistic regression", elastic_net = "Elastic net",
  decision_tree = "Decision tree", random_forest = "Random forest",
  svm_rbf = "RBF-SVM", knn = "KNN", lightgbm = "LightGBM",
  mlp = "MLP", xgboost = "XGBoost"
)
display_order <- c(
  "xgboost", "random_forest", "lightgbm", "svm_rbf", "logistic",
  "elastic_net", "knn", "mlp", "decision_tree"
)
if (!file.exists(nested_file)) {
  stop("Nested results RDS not found: ", nested_file)
}

# ---- 2. Authoritative OOF data -------------------------------------------
res <- readRDS(nested_file)
required_fields <- c(
  "all_model_patient_oof", "all_model_summary", "selected_model", "settings"
)
if (!all(required_fields %in% names(res))) {
  stop("Nested RDS lacks required OOF fields.")
}
if (!identical(as.character(res$selected_model), "random_forest")) {
  stop("The saved final algorithm is not random forest.")
}
repeats <- as.integer(as.character(
  res$settings$value[res$settings$setting == "outer_repeats"]
))
if (length(repeats) != 1L || is.na(repeats) || repeats != 5L) {
  stop("Expected five outer repeats.")
}
summary_data <- as.data.frame(res$all_model_summary)
summary_fields <- c("model", "completed_outer_folds", "complete_oof_coverage")
if (!all(summary_fields %in% names(summary_data)) ||
    nrow(summary_data) != 9L || anyDuplicated(summary_data$model) ||
    !setequal(as.character(summary_data$model), models) ||
    anyNA(summary_data[summary_fields]) ||
    any(summary_data$completed_outer_folds != 25L) ||
    !all(summary_data$complete_oof_coverage)) {
  stop("Complete nine-model, 25-assessment OOF coverage is required.")
}
patient_oof <- as.data.frame(res$all_model_patient_oof)
required_columns <- c(
  "patient_id", "model", "truth", "probability", "n_predictions"
)
if (!all(required_columns %in% names(patient_oof))) {
  stop("Required patient-level OOF columns are missing.")
}
patient_oof$patient_id <- as.character(patient_oof$patient_id)
patient_oof$model <- as.character(patient_oof$model)
patient_oof$truth <- as.character(patient_oof$truth)
patient_oof$probability <- as.numeric(patient_oof$probability)
if (anyNA(patient_oof[required_columns]) ||
    nrow(patient_oof) != 545L * 9L ||
    anyDuplicated(paste(patient_oof$model, patient_oof$patient_id, sep = "|")) ||
    !setequal(unique(patient_oof$model), models) ||
    !all(patient_oof$truth %in% c(positive, negative)) ||
    any(patient_oof$n_predictions != repeats) ||
    any(!is.finite(patient_oof$probability)) ||
    any(patient_oof$probability < 0 | patient_oof$probability > 1)) {
  stop("Invalid or incomplete patient-level OOF predictions.")
}
reference <- patient_oof[patient_oof$model == "random_forest", , drop = FALSE]
if (nrow(reference) != 545L || sum(reference$truth == positive) != 209L) {
  stop("Expected 545 development patients and 209 impacted-stone events.")
}
for (m in models) {
  d <- patient_oof[patient_oof$model == m, , drop = FALSE]
  if (nrow(d) != 545L || !setequal(d$patient_id, reference$patient_id)) {
    stop("Patient coverage differs for model: ", m)
  }
  k <- match(reference$patient_id, d$patient_id)
  if (!identical(reference$truth, d$truth[k])) {
    stop("Outcome labels differ across algorithms: ", m)
  }
}
# Preserve each model's original RDS row order during calculation.
# Sorting before bootstrap can change finite-resample percentile intervals.

# ---- 3. Metric functions aligned with Table 3 ----------------------------
safe_divide <- function(k, n) if (n > 0) k / n else NA_real_
exact_rate <- function(k, n) {
  if (n <= 0) return(c(estimate = NA_real_, low = NA_real_, high = NA_real_))
  ci <- stats::binom.test(
    as.integer(k), as.integer(n), conf.level = conf_level
  )$conf.int
  c(estimate = k / n, low = ci[[1L]], high = ci[[2L]])
}
counts <- function(y, p, t = threshold) {
  predicted <- p >= t
  c(TP = sum(y == 1L & predicted), FP = sum(y == 0L & predicted),
    TN = sum(y == 0L & !predicted), FN = sum(y == 1L & !predicted))
}
calibration <- function(y, p) {
  if (length(unique(y)) != 2L) {
    return(c(intercept = NA_real_, slope = NA_real_))
  }
  lp <- stats::qlogis(pmin(1 - 1e-6, pmax(1e-6, p)))
  if (stats::sd(lp) < 1e-10) {
    return(c(intercept = NA_real_, slope = NA_real_))
  }
  fit <- function(x, offset = NULL) {
    if (is.null(offset)) offset <- rep(0, length(y))
    ans <- suppressWarnings(tryCatch(
      stats::glm.fit(
        x = x, y = y, offset = offset, family = stats::binomial(),
        control = stats::glm.control(maxit = 50)
      ), error = function(e) NULL
    ))
    if (is.null(ans) || !isTRUE(ans$converged) ||
        any(!is.finite(ans$coefficients))) return(NULL)
    ans$coefficients
  }
  a <- fit(matrix(1, nrow = length(y), ncol = 1L), offset = lp)
  b <- fit(cbind(1, lp))
  c(intercept = if (is.null(a)) NA_real_ else a[[1L]],
    slope = if (is.null(b)) NA_real_ else b[[2L]])
}
derived_metrics <- function(y, p) {
  z <- counts(y, p)
  tp <- z[["TP"]]; fp <- z[["FP"]]
  tn <- z[["TN"]]; fn <- z[["FN"]]
  denominator <- sqrt(as.double(tp + fp) * (tp + fn) *
                        (tn + fp) * (tn + fn))
  cal <- calibration(y, p)
  c(F1 = safe_divide(2 * tp, 2 * tp + fp + fn),
    MCC = if (denominator > 0) (tp * tn - fp * fn) / denominator else NA_real_,
    Brier = mean((y - p)^2),
    cal_intercept = cal[["intercept"]], cal_slope = cal[["slope"]],
    Youden = safe_divide(tp, tp + fn) + safe_divide(tn, tn + fp) - 1)
}
bootstrap_interval <- function(x) {
  good <- x[is.finite(x)]
  if (length(good) < 0.8 * bootstrap_reps) return(c(NA_real_, NA_real_))
  as.numeric(stats::quantile(
    good, probs = c(0.025, 0.975), names = FALSE, type = 7
  ))
}
evaluate_model <- function(d, model_id, rng_seed) {
  y <- as.integer(d$truth == positive)
  p <- as.numeric(d$probability)
  z <- counts(y, p)
  tp <- z[["TP"]]; fp <- z[["FP"]]
  tn <- z[["TN"]]; fn <- z[["FN"]]
  roc <- pROC::roc(
    response = factor(y, levels = c(0, 1)), predictor = p,
    levels = c("0", "1"), direction = "<", quiet = TRUE
  )
  auc_ci <- as.numeric(pROC::ci.auc(
    roc, conf.level = conf_level, method = "delong"
  ))
  fixed <- list(
    AUC = c(estimate = as.numeric(pROC::auc(roc)),
            low = auc_ci[[1L]], high = auc_ci[[3L]]),
    Sensitivity = exact_rate(tp, tp + fn),
    Specificity = exact_rate(tn, tn + fp),
    PPV = exact_rate(tp, tp + fp), NPV = exact_rate(tn, tn + fn),
    Accuracy = exact_rate(tp + tn, length(y))
  )
  point <- derived_metrics(y, p)
  set.seed(rng_seed)
  boot <- replicate(bootstrap_reps, {
    i <- sample.int(length(y), size = length(y), replace = TRUE)
    derived_metrics(y[i], p[i])
  })
  boot <- matrix(boot, nrow = length(point),
                 dimnames = list(names(point), NULL))
  boot_ci <- t(apply(boot, 1L, bootstrap_interval))
  valid <- rowSums(is.finite(boot))
  if (any(valid < 0.8 * bootstrap_reps)) {
    warning("Some bootstrap intervals are unavailable for ", model_id,
            ". See valid bootstrap counts in the audit CSV.")
  }
  fixed_results <- do.call(rbind, lapply(names(fixed), function(nm) {
    a <- fixed[[nm]]
    data.frame(
      model = model_id, Algorithm = unname(model_labels[[model_id]]),
      metric = nm, estimate = unname(a[["estimate"]]),
      low = unname(a[["low"]]), high = unname(a[["high"]]),
      ci_method = if (nm == "AUC") "DeLong" else "Exact binomial",
      valid_bootstraps = NA_integer_, stringsAsFactors = FALSE
    )
  }))
  bootstrap_results <- do.call(rbind, lapply(names(point), function(nm) {
    data.frame(
      model = model_id, Algorithm = unname(model_labels[[model_id]]),
      metric = nm, estimate = unname(point[[nm]]),
      low = boot_ci[nm, 1L], high = boot_ci[nm, 2L],
      ci_method = "Ordinary patient bootstrap percentile",
      valid_bootstraps = valid[[nm]], stringsAsFactors = FALSE
    )
  }))
  list(metrics = rbind(fixed_results, bootstrap_results), counts = z,
       n = length(y), events = sum(y), seed = rng_seed)
}

# ---- 4. Calculate all nine algorithms ------------------------------------
fits <- setNames(lapply(seq_along(models), function(i) {
  m <- models[[i]]
  d <- patient_oof[patient_oof$model == m, , drop = FALSE]
  model_seed <- if (m == "random_forest") seed else seed + 100L + i
  message("Calculating: ", model_labels[[m]])
  evaluate_model(d, m, model_seed)
}), models)
numeric_results <- do.call(rbind, lapply(fits, function(x) x$metrics))
rownames(numeric_results) <- NULL

# ---- 5. Optional cross-table comparisons ---------------------------------
comparison_rows <- list()
compare_triplet <- function(model_id, metric_id, ref_values, reference_name) {
  current <- numeric_results[
    numeric_results$model == model_id & numeric_results$metric == metric_id,
    c("estimate", "low", "high"), drop = FALSE
  ]
  if (nrow(current) != 1L || length(ref_values) != 3L) {
    stop("Invalid comparison record: ", reference_name)
  }
  a <- as.numeric(unlist(current, use.names = FALSE))
  b <- as.numeric(ref_values)
  equal <- (is.na(a) & is.na(b)) |
    (is.finite(a) & is.finite(b) & abs(a - b) <= 1e-10)
  equal[is.na(equal)] <- FALSE
  comparison_rows[[length(comparison_rows) + 1L]] <<- data.frame(
    reference = reference_name, model = model_id, metric = metric_id,
    component = c("estimate", "low", "high"), current = a,
    reference_value = b, difference = a - b, match = equal,
    stringsAsFactors = FALSE
  )
}
if (file.exists(table2_audit_file)) {
  tab2 <- read.csv(table2_audit_file, check.names = FALSE)
  map2 <- c(AUC = "pooled_oof_auc", Brier = "pooled_oof_brier",
            cal_intercept = "calibration_intercept", cal_slope = "calibration_slope")
  needed <- c("model", unname(map2), paste0(unname(map2), "_lcl95"),
              paste0(unname(map2), "_ucl95"))
  if (!all(needed %in% names(tab2))) {
    stop("Table 2 audit CSV has an unexpected structure.")
  }
  for (m in models) {
    r <- tab2[tab2$model == m, , drop = FALSE]
    if (nrow(r) != 1L) stop("Table 2 audit lacks a unique row for: ", m)
    for (metric in names(map2)) {
      prefix <- map2[[metric]]
      values <- unlist(r[c(prefix, paste0(prefix, "_lcl95"),
                            paste0(prefix, "_ucl95"))], use.names = FALSE)
      compare_triplet(m, metric, values, "Table 2")
    }
  }
} else message("Table 2 audit CSV not found; cross-table check skipped.")
if (file.exists(table3_audit_file)) {
  tab3 <- read.csv(table3_audit_file, check.names = FALSE)
  needed <- c("cohort", "metric", "estimate", "low", "high")
  if (!all(needed %in% names(tab3))) {
    stop("Table 3 audit CSV has an unexpected structure.")
  }
  tab3 <- tab3[grepl("^Internal", tab3$cohort), , drop = FALSE]
  for (metric in unique(numeric_results$metric)) {
    if (metric == "Youden") next
    r <- tab3[tab3$metric == metric, , drop = FALSE]
    if (nrow(r) != 1L) stop("Table 3 audit lacks a unique internal metric: ", metric)
    compare_triplet("random_forest", metric,
                    unlist(r[c("estimate", "low", "high")], use.names = FALSE),
                    "Table 3 internal")
  }
} else message("Table 3 audit CSV not found; cross-table check skipped.")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (length(comparison_rows)) {
  comparison <- do.call(rbind, comparison_rows)
  write.csv(comparison, file.path(output_dir, "Table_S2_cross_table_comparison.csv"),
            row.names = FALSE)
  if (any(!comparison$match)) {
    print(comparison[!comparison$match, ], row.names = FALSE)
    stop("Cross-table disagreement detected. Check RDS, patient order, RNG settings ",
         "and reference audit versions. No new publication table was generated.")
  }
  message("All available cross-table checks passed.")
}

# ---- 6. Publication-format panels ----------------------------------------
format_metric <- function(model_id, metric_id) {
  r <- numeric_results[numeric_results$model == model_id &
                         numeric_results$metric == metric_id, , drop = FALSE]
  if (nrow(r) != 1L) stop("Missing metric: ", model_id, " / ", metric_id)
  if (!is.finite(r$estimate)) return("NE")
  if (!is.finite(r$low) || !is.finite(r$high)) {
    return(sprintf("%.3f (CI NE)", r$estimate))
  }
  sprintf("%.3f (%.3f\u2013%.3f)", r$estimate, r$low, r$high)
}
metric_column <- function(metric) {
  vapply(display_order, function(m) format_metric(m, metric), character(1))
}
count_text <- vapply(display_order, function(m) {
  paste(fits[[m]]$counts[c("TP", "FN", "TN", "FP")], collapse = " / ")
}, character(1))
# Three readable panels include the previously omitted Accuracy column.
panel_a <- data.frame(
  Algorithm = unname(model_labels[display_order]), Cutoff = sprintf("%.3f", threshold),
  N = vapply(display_order, function(m) fits[[m]]$n, integer(1)),
  `TP / FN / TN / FP` = unname(count_text),
  `Sensitivity (95% CI)` = metric_column("Sensitivity"),
  `Specificity (95% CI)` = metric_column("Specificity"),
  `PPV (95% CI)` = metric_column("PPV"), `NPV (95% CI)` = metric_column("NPV"),
  `Accuracy (95% CI)` = metric_column("Accuracy"), check.names = FALSE
)
panel_b <- data.frame(
  Algorithm = unname(model_labels[display_order]),
  `F1 score (95% CI)` = metric_column("F1"), `MCC (95% CI)` = metric_column("MCC"),
  `Youden index (95% CI)` = metric_column("Youden"), check.names = FALSE
)
panel_c <- data.frame(
  Algorithm = unname(model_labels[display_order]),
  `ROC AUC (95% CI)` = metric_column("AUC"),
  `Brier score (95% CI)` = metric_column("Brier"),
  `Calibration intercept (95% CI)` = metric_column("cal_intercept"),
  `Calibration slope (95% CI)` = metric_column("cal_slope"), check.names = FALSE
)

# ---- 7. Numeric outputs and provenance -----------------------------------
write.csv(numeric_results, file.path(output_dir, "Table_S2_numeric_audit.csv"),
          row.names = FALSE)
write.csv(patient_oof, file.path(output_dir,
          "Table_S2_patient_level_nested_OOF_predictions.csv"), row.names = FALSE)
count_audit <- do.call(rbind, lapply(models, function(m) {
  x <- fits[[m]]
  data.frame(model = m, N = x$n, events = x$events,
             TP = x$counts[["TP"]], FN = x$counts[["FN"]],
             TN = x$counts[["TN"]], FP = x$counts[["FP"]],
             threshold = threshold, bootstrap_seed = x$seed, row.names = NULL)
}))
write.csv(count_audit, file.path(output_dir, "Table_S2_counts_and_seeds.csv"),
          row.names = FALSE)
for (nm in c("a", "b", "c")) {
  write.csv(get(paste0("panel_", nm)),
            file.path(output_dir, paste0("Table_S2_part_", toupper(nm), ".csv")),
            row.names = FALSE, fileEncoding = "UTF-8")
}
write.csv(data.frame(
  setting = c("input_file", "input_md5", "bootstrap_reps", "RF_seed",
              "RNG_kind", "R_version", "pROC_version"),
  value = c(normalizePath(nested_file, winslash = "/"),
            unname(tools::md5sum(nested_file)), bootstrap_reps, seed,
            paste(RNGkind(), collapse = "; "), R.version.string,
            as.character(utils::packageVersion("pROC")))
), file.path(output_dir, "Table_S2_configuration.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()),
           file.path(output_dir, "Table_S2_sessionInfo.txt"))

# ---- 8. Editable Word document -------------------------------------------
caption <- paste(
  "The positive event was impacted stone for all nine algorithms.",
  "Patient-level OOF probabilities were averaged across five outer repeats",
  "before classification at the fixed threshold of 0.50.",
  "AUC confidence intervals used DeLong's method.",
  "Sensitivity, specificity, PPV, NPV and accuracy used exact binomial intervals.",
  "F1 score, MCC, Youden index, Brier score and calibration parameters used",
  "percentile intervals from 2,000 ordinary patient-level bootstrap resamples",
  "without outcome stratification.",
  "Calibration intercept used logit(predicted probability) as an offset;",
  "calibration slope estimated a free intercept and slope.",
  "Original probabilities were used for AUC and Brier calculations;",
  "clipping to [0.000001, 0.999999] was used only for calibration logits.",
  "Bootstrap intervals condition on saved predictions and do not account for",
  "uncertainty from repeating model development.",
  "The internal OOF predictions are not an independent test of the",
  "final full-development-cohort refitted model."
)
abbreviations <- paste(
  "AUC, area under the receiver operating characteristic curve;",
  "CI, confidence interval; MCC, Matthews correlation coefficient;",
  "PPV, positive predictive value; NPV, negative predictive value;",
  "TP, true positive; FN, false negative; TN, true negative;",
  "FP, false positive; NE, not estimable."
)
make_table <- function(d, widths) {
  ft <- flextable::flextable(d)
  ft <- flextable::theme_booktabs(ft)
  ft <- flextable::font(ft, fontname = "Arial", part = "all")
  ft <- flextable::fontsize(ft, size = 8.5, part = "all")
  ft <- flextable::bold(ft, part = "header")
  ft <- flextable::align(ft, align = "center", part = "all")
  ft <- flextable::align(ft, j = 1L, align = "left", part = "all")
  ft <- flextable::valign(ft, valign = "center", part = "all")
  ft <- flextable::padding(ft, padding = 3, part = "all")
  ft <- flextable::width(ft, j = seq_along(widths), width = widths)
  rf_row <- which(d$Algorithm == "Random forest")
  ft <- flextable::bold(ft, i = rf_row, part = "body")
  ft <- flextable::bg(ft, i = rf_row, bg = "#EEF2F5", part = "body")
  flextable::set_table_properties(
    ft, layout = "fixed", opts_word = list(split = FALSE)
  )
}
doc <- officer::read_docx()
doc <- officer::body_add_fpar(doc, officer::fpar(
  officer::ftext(
    "Table S2. Nine-algorithm comparison using patient-level nested OOF predictions",
    prop = officer::fp_text(font.family = "Arial", font.size = 11, bold = TRUE)
  ), fp_p = officer::fp_par(keep_with_next = TRUE)
))
panels <- list(panel_a, panel_b, panel_c)
headings <- c("Part A. Confusion counts and classification proportions",
              "Part B. Derived classification metrics",
              "Part C. Discrimination, probability error and calibration")
widths <- list(c(1.35, 0.50, 0.40, 1.10, 1.12, 1.12, 1.05, 1.05, 1.12),
               c(1.65, 1.65, 1.65, 1.75), c(1.60, 1.55, 1.55, 1.95, 1.85))
for (i in seq_along(panels)) {
  doc <- officer::body_add_fpar(doc, officer::fpar(
    officer::ftext(headings[[i]], prop = officer::fp_text(
      font.family = "Arial", font.size = 10, bold = TRUE
    )), fp_p = officer::fp_par(keep_with_next = TRUE)
  ))
  doc <- flextable::body_add_flextable(doc, make_table(panels[[i]], widths[[i]]))
  doc <- officer::body_add_par(doc, "")
}
for (text in c(caption, abbreviations)) {
  doc <- officer::body_add_fpar(doc, officer::fpar(officer::ftext(
    text, prop = officer::fp_text(font.family = "Arial", font.size = 8)
  )))
}
doc <- officer::body_set_default_section(doc, value = officer::prop_section(
  page_size = officer::page_size(orient = "landscape"),
  page_margins = officer::page_mar(top = 0.45, bottom = 0.45,
                                  left = 0.45, right = 0.45)
))
word_file <- file.path(output_dir, "Table_S2_nine_algorithms_unified_CI.docx")
print(doc, target = word_file)
writeLines(c(caption, abbreviations),
           file.path(output_dir, "Table_S2_caption_and_footnote.txt"),
           useBytes = TRUE)
message("Saved: ", word_file)
