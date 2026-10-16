# Table 2: nine-algorithm comparison with CI methods aligned to Table 3.
#
# Run:
# source("C:/Users/Lenovo/Desktop/impacted_pr/Table2_nested_algorithm_comparison_unified_CI.R")
#
# No model refitting or threshold optimization.
# Pooled OOF AUC: DeLong 95% CI.
# Pooled OOF Brier and calibration: 2,000 ordinary patient bootstrap resamples.
#
# RF uses the same seed and original patient order as Table 3.
# For exact bootstrap agreement, run both scripts with the same R RNG settings.

# ---- Paths and settings --------------------------------------------------
root <- "C:/Users/Lenovo/Desktop/impacted_pr"

results_file <- file.path(
  root, "output", "fully_nested_lasso_9models",
  "fully_nested_lasso_9models_results.rds"
)
bundle_file <- file.path(
  root, "output", "final_selected_model",
  "final_random_forest_model_bundle.rds"
)
output_dir <- file.path(
  root, "output", "Table2_nested_algorithm_comparison_unified_CI"
)

positive <- "Impacted"
bootstrap_reps <- 2000L
seed <- 20260926L
conf_level <- 0.95

models <- c(
  "logistic", "elastic_net", "decision_tree", "random_forest",
  "svm_rbf", "knn", "lightgbm", "mlp", "xgboost"
)

packages <- c("pROC", "flextable", "officer")
missing_packages <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Install required packages: ",
    paste(missing_packages, collapse = ", ")
  )
}
if (!file.exists(results_file) || !file.exists(bundle_file)) {
  stop("Missing nested results RDS or final RF bundle.")
}

# ---- Provenance ----------------------------------------------------------
res <- readRDS(results_file)
bundle <- readRDS(bundle_file)

required <- c(
  "all_model_summary", "all_model_fold_metrics",
  "all_model_patient_oof", "algorithm_selection_frequency",
  "selected_model", "settings"
)
if (!all(required %in% names(res))) {
  stop(
    "Nested results missing: ",
    paste(setdiff(required, names(res)), collapse = ", ")
  )
}
if (!identical(as.character(res$selected_model), "random_forest") ||
    !identical(as.character(bundle$selected_model), "random_forest")) {
  stop("The nested analysis and final bundle must identify random forest.")
}

config <- bundle$configuration
if (!is.data.frame(config) ||
    !all(c("setting", "value") %in% names(config))) {
  stop("The model bundle lacks provenance configuration.")
}
recorded_md5 <- as.character(
  config$value[config$setting == "nested_results_md5"]
)
if (length(recorded_md5) != 1L ||
    unname(tools::md5sum(results_file)) != recorded_md5) {
  stop("Nested results differ from the file used to lock the RF.")
}

# ---- Outer-fold and selection audits -------------------------------------
summ <- as.data.frame(res$all_model_summary)
fold <- as.data.frame(res$all_model_fold_metrics)
freq <- as.data.frame(res$algorithm_selection_frequency)

need_summary <- c(
  "model", "model_label", "outer_auc_mean", "outer_auc_sd",
  "outer_brier_mean", "completed_outer_folds", "expected_outer_folds",
  "failed_outer_folds", "n_patients", "complete_oof_coverage"
)
need_fold <- c("model", "outer_id", "roc_auc", "brier")
need_freq <- c("model", "selected_outer_folds", "selection_frequency")

if (!all(need_summary %in% names(summ)) ||
    !all(need_fold %in% names(fold)) ||
    !all(need_freq %in% names(freq))) {
  stop("Required outer-fold or selection audit fields are missing.")
}
if (nrow(summ) != 9L || nrow(freq) != 9L || nrow(fold) != 225L ||
    anyDuplicated(summ$model) || anyDuplicated(freq$model) ||
    !setequal(summ$model, models) ||
    !setequal(freq$model, models) ||
    !setequal(unique(fold$model), models) ||
    anyDuplicated(paste(fold$model, fold$outer_id, sep = "|"))) {
  stop("Unexpected or duplicated model/fold records.")
}

summary_numeric <- c(
  "outer_auc_mean", "outer_auc_sd", "outer_brier_mean",
  "completed_outer_folds", "expected_outer_folds",
  "failed_outer_folds", "n_patients"
)
if (any(!is.finite(as.matrix(summ[summary_numeric]))) ||
    any(!is.finite(fold$roc_auc)) ||
    any(!is.finite(fold$brier)) ||
    any(fold$roc_auc < 0 | fold$roc_auc > 1) ||
    any(fold$brier < 0 | fold$brier > 1) ||
    any(summ$expected_outer_folds != 25L) ||
    any(summ$completed_outer_folds != 25L) ||
    any(summ$failed_outer_folds != 0L) ||
    any(summ$n_patients != 545L) ||
    anyNA(summ$complete_oof_coverage) ||
    !all(summ$complete_oof_coverage)) {
  stop("Complete 25-fold, 545-patient validation is required.")
}

reference_folds <- sort(unique(as.character(
  fold$outer_id[fold$model == models[1L]]
)))

for (m in models) {
  f <- fold[fold$model == m, , drop = FALSE]
  s <- summ[summ$model == m, , drop = FALSE]

  if (nrow(f) != 25L ||
      length(reference_folds) != 25L ||
      !setequal(as.character(f$outer_id), reference_folds)) {
    stop("Outer-fold coverage differs across models: ", m)
  }

  differences <- c(
    mean(f$roc_auc) - s$outer_auc_mean,
    stats::sd(f$roc_auc) - s$outer_auc_sd,
    mean(f$brier) - s$outer_brier_mean
  )
  if (any(abs(differences) > 1e-8)) {
    stop("Saved summary differs from raw outer-fold metrics: ", m)
  }
}

if (any(!is.finite(freq$selected_outer_folds)) ||
    any(!is.finite(freq$selection_frequency)) ||
    any(freq$selected_outer_folds < 0L) ||
    any(freq$selected_outer_folds != floor(freq$selected_outer_folds)) ||
    sum(freq$selected_outer_folds) != 25L ||
    any(abs(
      freq$selected_outer_folds / 25 - freq$selection_frequency
    ) > 1e-10)) {
  stop("Invalid algorithm-selection frequencies.")
}

# ---- Patient-level OOF audit ---------------------------------------------
patient_oof <- as.data.frame(res$all_model_patient_oof)
need_patient <- c(
  "patient_id", "model", "truth", "probability", "n_predictions"
)
if (!all(need_patient %in% names(patient_oof))) {
  stop("Required patient-level OOF fields are missing.")
}

patient_oof$patient_id <- as.character(patient_oof$patient_id)
patient_oof$model <- as.character(patient_oof$model)
patient_oof$truth <- as.character(patient_oof$truth)

repeats <- as.integer(as.character(
  res$settings$value[res$settings$setting == "outer_repeats"]
))

if (length(repeats) != 1L || is.na(repeats) || repeats != 5L ||
    nrow(patient_oof) != 9L * 545L ||
    anyNA(patient_oof[need_patient]) ||
    anyDuplicated(paste(
      patient_oof$model, patient_oof$patient_id, sep = "|"
    )) ||
    !setequal(unique(patient_oof$model), models) ||
    any(patient_oof$n_predictions != repeats) ||
    !all(patient_oof$truth %in% c(positive, "Non-impacted"))) {
  stop("Patient-level OOF predictions are incomplete or invalid.")
}

# Preserve the original RF row order used by Table 3.
reference <- patient_oof[
  patient_oof$model == "random_forest", , drop = FALSE
]

for (m in models) {
  d <- patient_oof[patient_oof$model == m, , drop = FALSE]
  p <- as.numeric(d$probability)

  if (nrow(d) != 545L ||
      !setequal(d$patient_id, reference$patient_id) ||
      any(!is.finite(p) | p < 0 | p > 1)) {
    stop("Invalid patient coverage or probabilities: ", m)
  }

  k <- match(reference$patient_id, d$patient_id)
  if (!identical(reference$truth, d$truth[k])) {
    stop("Outcome coding differs across algorithms: ", m)
  }
}
if (sum(reference$truth == positive) != 209L) {
  stop("Expected 209 impacted-stone events among 545 patients.")
}

# ---- Calibration: same definition and fitting as Table 3 ------------------
calibration <- function(y, p) {
  if (length(unique(y)) != 2L) {
    return(c(intercept = NA_real_, slope = NA_real_))
  }

  # Clip only for the logit calculation.
  # AUC and Brier below always use original probabilities.
  lp <- stats::qlogis(pmin(1 - 1e-6, pmax(1e-6, p)))

  if (stats::sd(lp) < 1e-10) {
    return(c(intercept = NA_real_, slope = NA_real_))
  }

  fit <- function(x, offset = NULL) {
    if (is.null(offset)) offset <- rep(0, length(y))

    ans <- suppressWarnings(tryCatch(
      stats::glm.fit(
        x = x,
        y = y,
        offset = offset,
        family = stats::binomial(),
        control = stats::glm.control(maxit = 50)
      ),
      error = function(e) NULL
    ))

    if (is.null(ans) || !isTRUE(ans$converged) ||
        any(!is.finite(ans$coefficients))) {
      return(NULL)
    }
    ans$coefficients
  }

  a <- fit(matrix(1, nrow = length(y), ncol = 1), offset = lp)
  b <- fit(cbind(1, lp))

  c(
    intercept = if (is.null(a)) NA_real_ else a[[1L]],
    slope = if (is.null(b)) NA_real_ else b[[2L]]
  )
}

probability_metrics <- function(y, p) {
  cal <- calibration(y, p)
  c(
    pooled_oof_brier = mean((y - p)^2),
    calibration_intercept = cal[["intercept"]],
    calibration_slope = cal[["slope"]]
  )
}

bootstrap_interval <- function(values) {
  good <- values[is.finite(values)]
  if (length(good) < 0.8 * bootstrap_reps) {
    return(c(NA_real_, NA_real_))
  }
  as.numeric(stats::quantile(
    good,
    probs = c(0.025, 0.975),
    names = FALSE,
    type = 7
  ))
}

# ---- Pooled OOF estimates and CIs -----------------------------------------
evaluate_model <- function(d, model_name, rng_seed) {
  y <- as.integer(d$truth == positive)
  p <- as.numeric(d$probability)

  roc <- pROC::roc(
    response = factor(y, levels = c(0, 1)),
    predictor = p,
    levels = c("0", "1"),
    direction = "<",
    quiet = TRUE
  )
  auc_ci <- as.numeric(pROC::ci.auc(
    roc, conf.level = conf_level, method = "delong"
  ))

  point <- probability_metrics(y, p)

  # Same ordinary patient bootstrap as Table 3.
  # Adding F1/MCC calculations would not alter the sampled indices.
  set.seed(rng_seed)
  boot <- replicate(bootstrap_reps, {
    i <- sample.int(length(y), size = length(y), replace = TRUE)
    probability_metrics(y[i], p[i])
  })
  boot <- matrix(
    boot,
    nrow = length(point),
    dimnames = list(names(point), NULL)
  )
  boot_ci <- t(apply(boot, 1L, bootstrap_interval))
  valid <- rowSums(is.finite(boot))

  if (any(valid < 0.8 * bootstrap_reps)) {
    warning(
      "Insufficient valid bootstrap estimates for ", model_name,
      "; affected confidence intervals are unavailable."
    )
  }

  data.frame(
    model = model_name,
    pooled_oof_auc = as.numeric(pROC::auc(roc)),
    pooled_oof_auc_lcl95 = auc_ci[[1L]],
    pooled_oof_auc_ucl95 = auc_ci[[3L]],
    pooled_oof_brier = unname(point[["pooled_oof_brier"]]),
    pooled_oof_brier_lcl95 = boot_ci["pooled_oof_brier", 1L],
    pooled_oof_brier_ucl95 = boot_ci["pooled_oof_brier", 2L],
    calibration_intercept = unname(point[["calibration_intercept"]]),
    calibration_intercept_lcl95 = boot_ci["calibration_intercept", 1L],
    calibration_intercept_ucl95 = boot_ci["calibration_intercept", 2L],
    calibration_slope = unname(point[["calibration_slope"]]),
    calibration_slope_lcl95 = boot_ci["calibration_slope", 1L],
    calibration_slope_ucl95 = boot_ci["calibration_slope", 2L],
    valid_bootstraps_brier = valid[["pooled_oof_brier"]],
    valid_bootstraps_intercept = valid[["calibration_intercept"]],
    valid_bootstraps_slope = valid[["calibration_slope"]],
    bootstrap_seed = rng_seed,
    bootstrap_reps = bootstrap_reps,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

pooled <- do.call(rbind, lapply(seq_along(models), function(i) {
  m <- models[[i]]
  d <- patient_oof[patient_oof$model == m, , drop = FALSE]

  # RF must use precisely Table 3's internal seed.
  model_seed <- if (m == "random_forest") seed else seed + 100L + i

  evaluate_model(d, m, model_seed)
}))

if (any(!is.finite(pooled$pooled_oof_auc)) ||
    any(!is.finite(pooled$pooled_oof_brier))) {
  stop("A key pooled OOF metric could not be calculated.")
}

# ---- Publication table ---------------------------------------------------
row_order <- order(-summ$outer_auc_mean, match(summ$model, models))
summ <- summ[row_order, , drop = FALSE]
freq <- freq[match(summ$model, freq$model), , drop = FALSE]
pooled <- pooled[match(summ$model, pooled$model), , drop = FALSE]

if (!identical(as.character(summ$model), as.character(freq$model)) ||
    !identical(as.character(summ$model), as.character(pooled$model))) {
  stop("Could not align model summaries, frequencies and pooled estimates.")
}

raw <- data.frame(
  model = as.character(summ$model),
  Model = as.character(summ$model_label),
  outer_auc_mean = as.numeric(summ$outer_auc_mean),
  outer_auc_sd = as.numeric(summ$outer_auc_sd),
  outer_brier_mean = as.numeric(summ$outer_brier_mean),
  pooled[, setdiff(names(pooled), "model"), drop = FALSE],
  selected_outer_folds = as.integer(freq$selected_outer_folds),
  selection_frequency = as.numeric(freq$selection_frequency),
  completed_outer_folds = as.integer(summ$completed_outer_folds),
  expected_outer_folds = as.integer(summ$expected_outer_folds),
  check.names = FALSE,
  row.names = NULL
)

format_ci <- function(estimate, low, high) {
  vapply(seq_along(estimate), function(i) {
    if (!is.finite(estimate[[i]])) return("NE")
    if (!is.finite(low[[i]]) || !is.finite(high[[i]])) {
      return(sprintf("%.3f (CI NE)", estimate[[i]]))
    }
    sprintf("%.3f (%.3f–%.3f)", estimate[[i]], low[[i]], high[[i]])
  }, character(1))
}

tab <- data.frame(
  Model = raw$Model,
  `Outer AUC\nmean (SD)` = sprintf(
    "%.3f (%.3f)", raw$outer_auc_mean, raw$outer_auc_sd
  ),
  `Pooled OOF AUC\n(95% CI)` = format_ci(
    raw$pooled_oof_auc,
    raw$pooled_oof_auc_lcl95,
    raw$pooled_oof_auc_ucl95
  ),
  `Pooled OOF Brier\n(95% CI)` = format_ci(
    raw$pooled_oof_brier,
    raw$pooled_oof_brier_lcl95,
    raw$pooled_oof_brier_ucl95
  ),
  `Calibration intercept\n(95% CI)` = format_ci(
    raw$calibration_intercept,
    raw$calibration_intercept_lcl95,
    raw$calibration_intercept_ucl95
  ),
  `Calibration slope\n(95% CI)` = format_ci(
    raw$calibration_slope,
    raw$calibration_slope_lcl95,
    raw$calibration_slope_ucl95
  ),
  `Selected\nfolds, n (%)` = sprintf(
    "%d/25 (%.0f%%)",
    raw$selected_outer_folds,
    100 * raw$selection_frequency
  ),
  check.names = FALSE
)

rf_row <- which(raw$model == "random_forest")
if (nrow(tab) != 9L || ncol(tab) != 7L || length(rf_row) != 1L) {
  stop("Unexpected publication-table structure.")
}

title <- "Table 2. Comparison of nine algorithms in fully nested cross-validation"

notes <- paste(
  "Data: 545 development patients; 25 outer assessments per algorithm.",
  "Impacted stone is the positive event.",
  "Outer AUC mean (SD) summarizes the 25 outer assessments;",
  "the SD is descriptive and is not a confidence interval.",
  "Pooled OOF metrics use patient-level probabilities averaged across",
  "the five outer repeats.",
  "AUC 95% CIs use DeLong's method.",
  "Brier and calibration 95% CIs use 2,000 ordinary patient bootstrap",
  "resamples, without outcome stratification, and percentile limits.",
  "The bootstrap conditions on the saved OOF predictions and does not",
  "include uncertainty from refitting the complete nested procedure.",
  "Calibration intercept uses logit(predicted probability) as an offset;",
  "calibration slope estimates a free intercept and slope.",
  "Probabilities are clipped only for calibration logit calculations;",
  "AUC and Brier use original probabilities.",
  "Ideal calibration intercept and slope are 0 and 1, respectively.",
  "Selections use inner-training results only: models within 0.01 of",
  "the best AUC and 0.005 of the best Brier were assessed using",
  "calibration error (|intercept| + |slope - 1|, within 0.10 of the best);",
  "the prespecified complexity order resolved practical ties.",
  sprintf(
    "Random forest (shaded) was selected most often (%d/25);",
    raw$selected_outer_folds[rf_row]
  ),
  "the outer AUC ranking did not determine the final algorithm.",
  "RF pooled OOF methods and bootstrap seed are aligned with Table 3.",
  "AUC, area under the receiver operating characteristic curve;",
  "CI, confidence interval; OOF, out-of-fold; NE, not estimable."
)

# ---- Editable Word export ------------------------------------------------
ft <- flextable::flextable(tab)
ft <- flextable::theme_booktabs(ft)
ft <- flextable::set_table_properties(
  ft, layout = "fixed", opts_word = list(split = FALSE)
)
ft <- flextable::width(
  ft, j = seq_len(7L),
  width = c(1.55, 1.05, 1.28, 1.28, 1.35, 1.25, 1.05)
)
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 9, part = "all")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::valign(ft, valign = "center", part = "all")
ft <- flextable::align(ft, j = 1L, align = "left", part = "all")
ft <- flextable::align(ft, j = 2:7, align = "center", part = "all")
ft <- flextable::padding(
  ft, padding.top = 5, padding.bottom = 5,
  padding.left = 4, padding.right = 4, part = "all"
)
ft <- flextable::bold(ft, i = rf_row, part = "body")
ft <- flextable::bg(ft, i = rf_row, bg = "#EEF2F5", part = "body")

doc <- officer::read_docx()
doc <- officer::body_add_fpar(
  doc,
  officer::fpar(
    officer::ftext(
      title,
      prop = officer::fp_text(
        font.family = "Arial", font.size = 11, bold = TRUE
      )
    ),
    fp_p = officer::fp_par(keep_with_next = TRUE)
  )
)
doc <- flextable::body_add_flextable(doc, ft)
doc <- officer::body_add_fpar(
  doc,
  officer::fpar(
    officer::ftext(
      notes,
      prop = officer::fp_text(
        font.family = "Arial", font.size = 8.5, color = "#333333"
      )
    )
  )
)
doc <- officer::body_end_block_section(
  doc,
  officer::block_section(
    officer::prop_section(
      page_size = officer::page_size(orient = "landscape"),
      page_margins = officer::page_mar(
        top = 0.5, bottom = 0.5, left = 0.5, right = 0.5
      )
    )
  )
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(output_dir, "Table2_nested_algorithm_comparison_unified_CI")

print(doc, target = paste0(stem, ".docx"))
utils::write.csv(
  raw, paste0(stem, "_raw.csv"), row.names = FALSE
)
utils::write.csv(
  tab, paste0(stem, "_formatted.csv"), row.names = FALSE,
  fileEncoding = "UTF-8"
)
utils::write.csv(
  raw[rf_row, , drop = FALSE],
  paste0(stem, "_RF_internal_audit.csv"),
  row.names = FALSE
)
utils::write.csv(
  data.frame(
    setting = c(
      "nested_results_md5", "bootstrap_reps", "RF_bootstrap_seed",
      "RNG_kind", "R_version", "pROC_version",
      "AUC_CI_method", "bootstrap_method"
    ),
    value = c(
      recorded_md5, bootstrap_reps, seed,
      paste(RNGkind(), collapse = "; "),
      R.version.string,
      as.character(utils::packageVersion("pROC")),
      "DeLong",
      "Ordinary patient bootstrap percentile; no outcome stratification"
    )
  ),
  paste0(stem, "_configuration.csv"),
  row.names = FALSE
)
writeLines(notes, paste0(stem, "_footnote.txt"), useBytes = TRUE)

message("Saved Table 2 and audit outputs to: ", output_dir)