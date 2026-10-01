# Table 3 | Internal nested RF OOF and external locked-RF performance
# Run: source("C:/Users/Lenovo/Desktop/impacted_pr/Table3_final_RF_performance.R")
# Or: Rscript Table3_final_RF_performance.R [nested.rds] [bundle.rds]
#                                      [development.rds] [external.csv] [output_dir]
# The internal column is NOT an independent test of the refitted final RF.

root <- "C:/Users/Lenovo/Desktop/impacted_pr"
args <- commandArgs(trailingOnly = TRUE)
arg_or <- function(i, default) if (length(args) >= i) args[[i]] else default
nested_path <- arg_or(1L, file.path(root, "output", "fully_nested_lasso_9models",
                                    "fully_nested_lasso_9models_results.rds"))
bundle_path <- arg_or(2L, file.path(root, "output", "final_selected_model",
                                    "final_random_forest_model_bundle.rds"))
development_path <- arg_or(3L, file.path(root, "data", "impacted_clean.rds"))
external_path <- arg_or(4L, file.path(root, "data", "data_wb.csv"))
output_dir <- arg_or(5L, file.path(root, "output", "Table3_final_RF_performance"))

# Used only when the model bundle has no saved classification threshold.
# This is an analysis-defined 0.50 cutoff, not a retrospectively "locked" one.
fallback_threshold <- 0.50
bootstrap_reps <- 2000L
seed <- 20260926L

packages <- c("pROC", "workflows", "flextable", "officer")
missing <- packages[!vapply(packages, requireNamespace, logical(1),
                             quietly = TRUE)]
if (length(missing)) {
  stop("Install required packages first: install.packages(c(",
       paste(sprintf('"%s"', missing), collapse = ", "), "))")
}
paths <- c(nested_path, bundle_path, development_path, external_path)
if (any(!file.exists(paths))) {
  stop("Missing input file(s):\n", paste(paths[!file.exists(paths)], collapse = "\n"))
}

# ---- Data provenance and unmodified external predictions -----------------
bundle <- readRDS(bundle_path)
nested <- readRDS(nested_path)
if (!all(c("fitted_workflow", "selected_model", "selected_predictors",
           "configuration") %in% names(bundle)) ||
    !identical(as.character(bundle$selected_model), "random_forest") ||
    !inherits(bundle$fitted_workflow, "workflow") ||
    !identical(as.character(nested$selected_model), "random_forest")) {
  stop("The nested result and locked bundle must identify the same final RF.")
}
config <- bundle$configuration
if (!is.data.frame(config) || !all(c("setting", "value") %in% names(config))) {
  stop("The RF bundle is missing its provenance configuration.")
}
get_setting <- function(key) as.character(config$value[config$setting == key])
if (length(get_setting("data_md5")) != 1L ||
    length(get_setting("nested_results_md5")) != 1L ||
    unname(tools::md5sum(development_path)) != get_setting("data_md5") ||
    unname(tools::md5sum(nested_path)) != get_setting("nested_results_md5")) {
  stop("The development or nested results file differs from the RF lock inputs.")
}
predictors <- as.character(bundle$selected_predictors)
expected <- c("flank_pain", "stone_attenuation_hu", "proximal_ureter_width",
              "ureteral_wall_thickness", "stone_length", "hydronephrosis_area")
if (length(predictors) != 6L || !setequal(predictors, expected)) {
  stop("The locked RF predictors differ from the expected six predictors.")
}

raw_names <- c("Flank Pain" = "flank_pain", "Stone HU" = "stone_attenuation_hu",
               "WUPC" = "proximal_ureter_width", "UWT" = "ureteral_wall_thickness",
               "Stone length" = "stone_length", "APH" = "hydronephrosis_area",
               "Impacted stone" = "impacted_stone", "Name" = "patient_id")
prepare <- function(d, label) {
  d <- as.data.frame(d, check.names = FALSE)
  for (from in names(raw_names)) {
    to <- raw_names[[from]]
    if (from %in% names(d) && !to %in% names(d)) {
      names(d)[names(d) == from] <- to
    }
  }
  needed <- c(predictors, "impacted_stone")
  if (!nrow(d) || !all(needed %in% names(d)) || anyNA(d[needed])) {
    stop(label, " has missing columns, missing values, or no patients.")
  }
  for (nm in setdiff(predictors, "flank_pain")) {
    d[[nm]] <- suppressWarnings(as.numeric(as.character(d[[nm]])))
    if (any(!is.finite(d[[nm]]))) stop(label, " invalid numeric input: ", nm)
  }
  pain <- trimws(as.character(d$flank_pain))
  if (!all(pain %in% c("No", "Yes"))) stop(label, " invalid flank_pain.")
  d$flank_pain <- factor(pain, levels = c("No", "Yes"))
  truth <- trimws(as.character(d$impacted_stone))
  truth[truth == "Yes"] <- "Impacted"
  truth[truth == "No"] <- "Non-impacted"
  if (!all(truth %in% c("Impacted", "Non-impacted"))) {
    stop(label, " invalid impacted_stone coding.")
  }
  d$impacted_stone <- factor(truth, levels = c("Impacted", "Non-impacted"))
  if (any(table(d$impacted_stone) < 5L)) {
    stop(label, " needs at least five events and five non-events.")
  }
  d
}
development <- prepare(readRDS(development_path), "Development")
external <- prepare(read.csv(external_path, check.names = FALSE,
                             stringsAsFactors = FALSE), "External")
if (!"patient_id" %in% names(development) ||
    anyDuplicated(development$patient_id)) {
  stop("Development patient IDs are missing or duplicated.")
}
row_signature <- function(x) {
  apply(x[c(predictors, "impacted_stone")], 1L, paste, collapse = "|")
}
if (any(row_signature(external) %in% row_signature(development))) {
  stop("External records match records in the development dataset.")
}

needed_nested <- c("all_model_patient_oof", "all_model_summary", "settings")
if (!all(needed_nested %in% names(nested))) stop("Missing nested OOF results.")
oof <- as.data.frame(nested$all_model_patient_oof)
if (!all(c("patient_id", "model", "truth", "probability", "n_predictions") %in%
         names(oof))) stop("Nested OOF columns are incomplete.")
oof <- oof[as.character(oof$model) == "random_forest", , drop = FALSE]
repeats <- as.integer(as.character(nested$settings$value[
  nested$settings$setting == "outer_repeats"]))
completed <- nested$all_model_summary$completed_outer_folds[
  nested$all_model_summary$model == "random_forest"]
if (length(repeats) != 1L || length(completed) != 1L ||
    completed != 25L || nrow(oof) != nrow(development) ||
    anyDuplicated(oof$patient_id) ||
    !all(oof$n_predictions == repeats) ||
    !setequal(as.character(oof$patient_id),
              as.character(development$patient_id))) {
  stop("The RF candidate does not have complete patient-level nested OOF data.")
}
id_match <- match(as.character(oof$patient_id),
                  as.character(development$patient_id))
if (!identical(as.character(oof$truth),
               as.character(development$impacted_stone[id_match]))) {
  stop("Nested OOF and development outcomes do not agree.")
}
internal <- data.frame(y = as.integer(as.character(oof$truth) == "Impacted"),
                       p = as.numeric(oof$probability))
external_pred <- predict(bundle$fitted_workflow,
                         new_data = external[, predictors, drop = FALSE],
                         type = "prob")
if (!".pred_Impacted" %in% names(external_pred)) {
  stop("The locked RF did not return the probability of 'Impacted'.")
}
external_eval <- data.frame(y = as.integer(external$impacted_stone == "Impacted"),
                            p = as.numeric(external_pred$.pred_Impacted))
for (d in list(internal, external_eval)) {
  if (any(!is.finite(d$p) | d$p < 0 | d$p > 1)) {
    stop("Predicted event probabilities must be finite and within [0, 1].")
  }
}

# Read a previously saved cutoff if present. Never choose it on external data.
threshold_keys <- c("classification_threshold", "decision_threshold",
                    "probability_threshold", "fixed_threshold")
stored <- suppressWarnings(as.numeric(as.character(
  config$value[config$setting %in% threshold_keys])))
for (field in threshold_keys) {
  if (field %in% names(bundle)) {
    stored <- c(stored, suppressWarnings(as.numeric(bundle[[field]])))
  }
}
stored <- unique(stored[is.finite(stored)])
if (length(stored) > 1L ||
    (length(stored) && (stored <= 0 || stored >= 1))) {
  stop("The locked bundle contains conflicting or invalid thresholds.")
}
threshold <- if (length(stored)) stored else fallback_threshold
threshold_source <- if (length(stored)) "saved in locked bundle" else
  "analysis-defined 0.50; no classification threshold saved in bundle"

# ---- Point estimates and conditional 95% CIs -----------------------------
counts <- function(y, p, t) {
  pred <- p >= t
  c(TP = sum(y == 1L & pred), FP = sum(y == 0L & pred),
    TN = sum(y == 0L & !pred), FN = sum(y == 1L & !pred))
}
rate <- function(k, n) if (n > 0) k / n else NA_real_
exact_rate <- function(k, n) {
  if (!n) return(c(estimate = NA_real_, low = NA_real_, high = NA_real_))
  ci <- stats::binom.test(as.integer(k), as.integer(n), conf.level = 0.95)$conf.int
  c(estimate = k / n, low = ci[[1L]], high = ci[[2L]])
}
calibration <- function(y, p) {
  if (length(unique(y)) != 2L) return(c(intercept = NA_real_, slope = NA_real_))
  lp <- stats::qlogis(pmin(1 - 1e-6, pmax(1e-6, p)))
  if (stats::sd(lp) < 1e-10) {
    return(c(intercept = NA_real_, slope = NA_real_))
  }
  fit <- function(x, offset = NULL) {
    if (is.null(offset)) offset <- rep(0, length(y))
    ans <- suppressWarnings(tryCatch(
      stats::glm.fit(x = x, y = y, offset = offset,
                     family = stats::binomial(),
                     control = stats::glm.control(maxit = 50)),
      error = function(e) NULL))
    if (is.null(ans) || !isTRUE(ans$converged) ||
        any(!is.finite(ans$coefficients))) return(NULL)
    ans$coefficients
  }
  a <- fit(matrix(1, nrow = length(y), ncol = 1), offset = lp)
  b <- fit(cbind(1, lp))
  c(intercept = if (is.null(a)) NA_real_ else a[[1L]],
    slope = if (is.null(b)) NA_real_ else b[[2L]])
}
boot_metrics <- function(y, p, t) {
  z <- counts(y, p, t)
  tp <- z[["TP"]]; fp <- z[["FP"]]
  tn <- z[["TN"]]; fn <- z[["FN"]]
  denom <- sqrt(as.double(tp + fp) * (tp + fn) *
                  (tn + fp) * (tn + fn))
  cal <- calibration(y, p)
  c(F1 = rate(2 * tp, 2 * tp + fp + fn),
    MCC = if (denom > 0) (tp * tn - fp * fn) / denom else NA_real_,
    Brier = mean((y - p)^2),
    cal_intercept = cal[["intercept"]], cal_slope = cal[["slope"]])
}
bootstrap_interval <- function(values) {
  good <- values[is.finite(values)]
  if (length(good) < 0.8 * bootstrap_reps) return(c(NA_real_, NA_real_))
  as.numeric(stats::quantile(good, probs = c(0.025, 0.975), names = FALSE))
}
evaluate <- function(d, label, rng_seed) {
  y <- d$y; p <- d$p
  z <- counts(y, p, threshold)
  tp <- z[["TP"]]; fp <- z[["FP"]]
  tn <- z[["TN"]]; fn <- z[["FN"]]
  roc <- pROC::roc(response = factor(y, levels = c(0, 1)), predictor = p,
                   levels = c("0", "1"), direction = "<", quiet = TRUE)
  auc_ci <- as.numeric(pROC::ci.auc(roc, conf.level = 0.95,
                                   method = "delong"))
  fixed <- list(
    AUC = c(estimate = as.numeric(pROC::auc(roc)), low = auc_ci[[1L]],
            high = auc_ci[[3L]]),
    Sensitivity = exact_rate(tp, tp + fn),
    Specificity = exact_rate(tn, tn + fp),
    PPV = exact_rate(tp, tp + fp),
    NPV = exact_rate(tn, tn + fn),
    Accuracy = exact_rate(tp + tn, length(y))
  )
  point <- boot_metrics(y, p, threshold)
  set.seed(rng_seed)
  boot <- replicate(bootstrap_reps, {
    i <- sample.int(length(y), size = length(y), replace = TRUE)
    boot_metrics(y[i], p[i], threshold)
  })
  boot <- matrix(boot, nrow = length(point), dimnames = list(names(point), NULL))
  boot_ci <- t(apply(boot, 1L, bootstrap_interval))
  out <- do.call(rbind, lapply(names(fixed), function(nm) {
    x <- fixed[[nm]]
    data.frame(cohort = label, metric = nm, estimate = unname(x[["estimate"]]),
               low = unname(x[["low"]]), high = unname(x[["high"]]),
               ci_method = if (nm == "AUC") "DeLong" else "Exact binomial",
               valid_bootstraps = NA_integer_)
  }))
  for (nm in names(point)) {
    out <- rbind(out, data.frame(
      cohort = label, metric = nm, estimate = unname(point[[nm]]),
      low = boot_ci[nm, 1L], high = boot_ci[nm, 2L],
      ci_method = "Patient bootstrap percentile",
      valid_bootstraps = sum(is.finite(boot[nm, ]))))
  }
  list(metrics = out, counts = z, n = length(y), events = sum(y))
}
internal_res <- evaluate(internal, "Internal nested OOF RF candidates", seed)
external_res <- evaluate(external_eval, "External locked RF", seed + 1L)
numeric_results <- rbind(internal_res$metrics, external_res$metrics)
metric_order <- c("AUC", "Sensitivity", "Specificity", "PPV", "NPV",
                  "Accuracy", "F1", "MCC", "Brier", "cal_intercept", "cal_slope")
if (any(!is.finite(numeric_results$estimate[
    numeric_results$metric %in% c("AUC", "Brier", "Sensitivity",
                                  "Specificity", "Accuracy")]))) {
  stop("A key performance metric could not be calculated.")
}

# ---- Publication table: two cohorts in parallel --------------------------
format_ci <- function(r) {
  if (!is.finite(r$estimate)) return("NE")
  if (!is.finite(r$low) || !is.finite(r$high)) {
    return(sprintf("%.3f (CI NE)", r$estimate))
  }
  sprintf("%.3f (%.3f-%.3f)", r$estimate, r$low, r$high)
}
cell <- function(result, metric) {
  r <- result$metrics[result$metrics$metric == metric, , drop = FALSE]
  if (nrow(r) != 1L) stop("Missing metric: ", metric)
  format_ci(r)
}
row_titles <- c(
  "Patients, n", "Impacted stones, n (%)", "Classification threshold",
  "ROC AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy",
  "F1 score", "Matthews correlation coefficient (MCC)", "Brier score",
  "Calibration intercept", "Calibration slope", "TP / FP / TN / FN"
)
make_column <- function(result) {
  c(sprintf("%d", result$n),
    sprintf("%d (%.1f%%)", result$events, 100 * result$events / result$n),
    sprintf("%.3f", threshold),
    vapply(metric_order, function(k) cell(result, k), character(1)),
    paste(result$counts[c("TP", "FP", "TN", "FN")], collapse = " / "))
}
tab <- data.frame(
  Measure = row_titles,
  Internal = make_column(internal_res),
  External = make_column(external_res),
  check.names = FALSE, stringsAsFactors = FALSE
)
if (nrow(tab) != length(row_titles)) stop("The table has missing rows.")

subtitle <- sprintf("Event: impacted stone  |  Classification threshold: %.3f (%s)",
                    threshold, threshold_source)
notes <- c(
  "Internal: patient-level probabilities from the RF candidate in nested CV, averaged across outer repeats; these are not predictions from the refitted final RF or an independent internal test cohort.",
  "External: unchanged predictions of the locked final RF in independent patients. The same event definition and classification threshold are applied to both cohorts.",
  "Values are estimates (95% CI). AUC: DeLong; sensitivity, specificity, PPV, NPV, accuracy: exact binomial; F1, MCC, Brier and calibration: 2,000 patient bootstrap resamples.",
  "Calibration intercept fits logit(predicted probability) as an offset; calibration slope estimates a free intercept and slope. Original probabilities are used without recalibration.",
  "Bootstrap intervals condition on the saved predictions. The internal OOF intervals do not include uncertainty from re-running nested model development; external calibration may be imprecise.",
  "Positive class: impacted stone. TP/FP/TN/FN are counts at the stated cutoff. NE = not estimable; an unavailable CI is marked CI NE."
)

draw_table <- function() {
  grid::grid.newpage()
  ink <- "#263B49"; blue <- "#236A8A"; muted <- "#53626A"
  grid::grid.rect(gp = grid::gpar(fill = "white", col = NA))
  grid::grid.text("Table 3. Final RF performance and nested internal assessment",
                  x = 0.055, y = 0.96, just = "left",
                  gp = grid::gpar(fontsize = 14.5, fontface = "bold", col = ink))
  grid::grid.text(sprintf("Event: impacted stone  |  Cutoff %.3f (%s)",
                          threshold,
                          if (length(stored)) "bundle-locked" else "analysis-defined"),
                  x = 0.055, y = 0.921, just = "left",
                  gp = grid::gpar(fontsize = 9.5, col = muted))
  grid::grid.lines(x = c(.055, .945), y = c(.895, .895),
                   gp = grid::gpar(col = blue, lwd = 1.2))
  xs <- c(.062, .546, .815)
  grid::grid.text("Measure", x = xs[[1L]], y = .857, just = "left",
                  gp = grid::gpar(fontsize = 9.8, fontface = "bold", col = ink))
  grid::grid.text("Internal: nested OOF\nRF candidates", x = xs[[2L]],
                  y = .857, just = "center",
                  gp = grid::gpar(fontsize = 9.4, fontface = "bold",
                                  col = ink, lineheight = 1.05))
  grid::grid.text("External: locked\nfinal RF", x = xs[[3L]],
                  y = .857, just = "center",
                  gp = grid::gpar(fontsize = 9.4, fontface = "bold",
                                  col = ink, lineheight = 1.05))
  grid::grid.lines(x = c(.055, .945), y = c(.815, .815),
                   gp = grid::gpar(col = "#AFC0C9", lwd = .8))
  for (i in seq_len(nrow(tab))) {
    yy <- .792 - (i - 1L) * .0425
    if (i %% 2L == 0L || i %in% c(4L, 12L)) {
      grid::grid.rect(x = .5, y = yy, width = .89, height = .041,
                      gp = grid::gpar(fill = if (i %in% c(4L, 12L))
                        "#EAF2F6" else "#F5F8F9", col = NA))
    }
    for (j in seq_len(ncol(tab))) {
      grid::grid.text(tab[[j]][[i]], x = xs[[j]], y = yy,
                      just = if (j == 1L) "left" else "center",
                      gp = grid::gpar(fontsize = if (j == 1L) 9.0 else 8.8,
                                      fontface = if (i %in% c(4L, 12L)) "bold"
                                      else "plain", col = ink))
    }
  }
  grid::grid.lines(x = c(.055, .945), y = c(.145, .145),
                   gp = grid::gpar(col = "#AFC0C9", lwd = .8))
  short_notes <- c(
    "Internal = nested patient-level RF OOF; external = locked final RF. The two columns evaluate different fitted models.",
    "95% CI: DeLong (AUC), exact binomial (classification rates), patient bootstrap (F1/MCC/Brier/calibration).",
    "CI conditions on saved predictions; internal OOF is not an independent test. TP/FP/TN/FN: confusion counts.",
    if (length(stored)) "The threshold was read from the locked bundle."
    else "Threshold 0.50 was defined for this analysis; it was not stored as a locked clinical cutoff."
  )
  for (k in seq_along(short_notes)) {
    grid::grid.text(short_notes[[k]], x = .055, y = .12 - (k - 1L) * .026,
                    just = "left", gp = grid::gpar(fontsize = 7.2, col = muted))
  }
}

# ---- Editable Word document and publication exports -----------------------
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(output_dir, "Table3_final_RF_performance")
write.csv(tab, paste0(stem, "_formatted.csv"), row.names = FALSE,
          fileEncoding = "UTF-8")
write.csv(numeric_results, paste0(stem, "_numeric_audit.csv"), row.names = FALSE)
write.csv(data.frame(cohort = c("Internal nested OOF RF candidates",
                                "External locked RF"),
                     n = c(internal_res$n, external_res$n),
                     events = c(internal_res$events, external_res$events),
                     TP = c(internal_res$counts[["TP"]], external_res$counts[["TP"]]),
                     FP = c(internal_res$counts[["FP"]], external_res$counts[["FP"]]),
                     TN = c(internal_res$counts[["TN"]], external_res$counts[["TN"]]),
                     FN = c(internal_res$counts[["FN"]], external_res$counts[["FN"]]),
                     threshold = threshold, threshold_source = threshold_source),
          paste0(stem, "_counts.csv"), row.names = FALSE)
writeLines(c("Table 3. Final RF performance and nested internal assessment.",
             subtitle, notes), paste0(stem, "_legend.txt"), useBytes = TRUE)

ft <- flextable::flextable(tab)
ft <- flextable::set_header_labels(ft,
  Measure = "Measure", Internal = "Internal: nested OOF\nRF candidates",
  External = "External: locked\nfinal RF")
ft <- flextable::add_header_lines(ft, values = subtitle)
ft <- flextable::add_header_lines(
  ft, values = "Table 3. Final RF performance and nested internal assessment")
ft <- flextable::add_footer_lines(ft, values = notes)
ft <- flextable::border_remove(ft)
ft <- flextable::width(ft, j = names(tab), width = c(2.36, 2.33, 2.33))
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 9.0, part = "body")
ft <- flextable::fontsize(ft, size = 9.0, part = "header")
ft <- flextable::fontsize(ft, size = 7.8, part = "footer")
ft <- flextable::fontsize(ft, i = 1L, size = 12.5, part = "header")
ft <- flextable::fontsize(ft, i = 2L, size = 8.6, part = "header")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::bg(ft, part = "header", bg = "#E7F0F4")
ft <- flextable::bg(ft, i = seq(2L, nrow(tab), by = 2L),
                    part = "body", bg = "#F5F8F9")
ft <- flextable::bg(ft, i = c(4L, 12L), part = "body", bg = "#EAF2F6")
ft <- flextable::bold(ft, i = c(4L, 12L), part = "body")
ft <- flextable::align(ft, align = "center", part = "all")
ft <- flextable::align(ft, j = "Measure", align = "left", part = "body")
ft <- flextable::align(ft, align = "left", part = "footer")
ft <- flextable::valign(ft, valign = "center", part = "all")
ft <- flextable::padding(ft, padding = 3, part = "body")
ft <- flextable::padding(ft, padding = 2, part = "footer")
ft <- flextable::hline_bottom(
  ft, part = "header", border = officer::fp_border(color = "#AFC0C9", width = 1))
ft <- flextable::hline_top(
  ft, part = "footer", border = officer::fp_border(color = "#AFC0C9", width = 1))
ft <- flextable::set_table_properties(ft, layout = "fixed", width = 1,
                                      opts_word = list(split = FALSE))
section <- officer::prop_section(
  page_size = officer::page_size(orient = "portrait"),
  page_margins = officer::page_mar(top = .48, bottom = .48,
                                   left = .52, right = .52))
flextable::save_as_docx(ft, path = paste0(stem, ".docx"),
                        pr_section = section, align = "center")

pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
export_table <- function(path, open_device) {
  open_device(path)
  on.exit(grDevices::dev.off(), add = TRUE)
  draw_table()
}
export_table(paste0(stem, ".pdf"), function(path)
  pdf_device(path, width = 9.5, height = 10.2, family = "sans", bg = "white"))
export_table(paste0(stem, ".tiff"), function(path)
  grDevices::tiff(path, width = 9.5, height = 10.2, units = "in",
                 res = 600, compression = "lzw", bg = "white"))
export_table(paste0(stem, ".png"), function(path)
  grDevices::png(path, width = 9.5, height = 10.2, units = "in",
                res = 300, bg = "white"))
message("Saved Table 3: ", stem, " [DOCX, PDF, TIFF, PNG, CSV, legend]")
