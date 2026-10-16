# Figure 3 | RF nested-CV internal validation + locked RF external validation
# Left: development-cohort RF candidate, out-of-fold predictions only.
# Right: final locked RF applied unchanged to the external cohort.
# These columns describe DIFFERENT fits; never label the left as an
# independent test of the final refitted RF model.

# ---- 1. Paths and clinically prespecified choices -------------------------
project_dir <- "C:/Users/Lenovo/Desktop/impacted_pr"
bundle_file <- file.path(project_dir, "output", "final_selected_model",
                         "final_random_forest_model_bundle.rds")
development_file <- file.path(project_dir, "data", "impacted_clean.rds")
nested_file <- file.path(project_dir, "output", "fully_nested_lasso_9models",
                         "fully_nested_lasso_9models_results.rds")
external_file <- file.path(project_dir, "data", "data_wb.csv")
output_dir <- file.path(project_dir, "output", "Figure3_RF_nested_OOF_external")

# Threshold probability means the minimum risk of impaction at which a
# prespecified clinical action would be taken. Confirm the action and range
# with the clinical team before interpreting the decision curves.
thresholds <- seq(0.10, 0.60, by = 0.01)
bootstrap_reps <- 1000L
seed <- 20260926L

# ---- 2. Dependencies and provenance checks --------------------------------
packages <- c("ggplot2", "patchwork", "pROC", "workflows")
missing_packages <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Install packages: ", paste(missing_packages, collapse = ", "))
}

needed_files <- c(bundle_file, development_file, nested_file, external_file)
missing_files <- needed_files[!file.exists(needed_files)]
if (length(missing_files)) {
  stop("Required files are missing:\n", paste(missing_files, collapse = "\n"),
       "\nThe original analysis writes nested results to its output folder.")
}

bundle <- readRDS(bundle_file)
required_bundle <- c("fitted_workflow", "selected_model", "selected_predictors",
                     "configuration", "outcome")
if (!all(required_bundle %in% names(bundle)) ||
    !identical(as.character(bundle$selected_model), "random_forest") ||
    !inherits(bundle$fitted_workflow, "workflow")) {
  stop("The locked random-forest workflow/bundle is invalid.")
}
predictors <- as.character(bundle$selected_predictors)
expected_predictors <- c(
  "flank_pain", "stone_attenuation_hu", "proximal_ureter_width",
  "ureteral_wall_thickness", "stone_length", "hydronephrosis_area"
)
if (!setequal(predictors, expected_predictors) || length(predictors) != 6L) {
  stop("The six predictors do not match the locked RF bundle.")
}

saved_md5 <- as.character(bundle$configuration$value[
  bundle$configuration$setting == "data_md5"
])
if (length(saved_md5) != 1L ||
    unname(tools::md5sum(development_file)) != saved_md5) {
  stop("Development data MD5 differs from the data used to fit this RF bundle.")
}
nested_md5 <- as.character(bundle$configuration$value[
  bundle$configuration$setting == "nested_results_md5"
])
if (length(nested_md5) != 1L ||
    unname(tools::md5sum(nested_file)) != nested_md5) {
  stop("Nested-CV results differ from those used to select the locked RF.")
}

read_cohort <- function(path) {
  if (grepl("[.]rds$", path, ignore.case = TRUE)) {
    as.data.frame(readRDS(path), check.names = FALSE)
  } else if (grepl("[.]csv$", path, ignore.case = TRUE)) {
    read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  } else {
    stop("Only .rds and .csv cohorts are supported: ", path)
  }
}

# Map the source's raw CT/clinical names to the locked model's names.
raw_to_model <- c(
  "Flank Pain" = "flank_pain",
  "Stone HU" = "stone_attenuation_hu",
  "WUPC" = "proximal_ureter_width",
  "UWT" = "ureteral_wall_thickness",
  "Stone length" = "stone_length",
  "APH" = "hydronephrosis_area",
  "Impacted stone" = "impacted_stone",
  "Name" = "patient_id"
)

prepare_cohort <- function(x, label) {
  for (raw_name in names(raw_to_model)) {
    clean_name <- raw_to_model[[raw_name]]
    if (raw_name %in% names(x) && !clean_name %in% names(x)) {
      names(x)[names(x) == raw_name] <- clean_name
    }
  }
  required <- c(predictors, "impacted_stone")
  if (!all(required %in% names(x))) {
    stop(label, " missing columns: ",
         paste(setdiff(required, names(x)), collapse = ", "))
  }
  if (!nrow(x) || anyNA(x[required])) stop(label, " has no rows or missing inputs.")

  for (name in setdiff(predictors, "flank_pain")) {
    value <- suppressWarnings(as.numeric(as.character(x[[name]])))
    if (any(!is.finite(value))) stop(label, " invalid numeric predictor: ", name)
    x[[name]] <- value
  }
  pain <- trimws(as.character(x$flank_pain))
  if (!all(pain %in% c("No", "Yes"))) {
    stop(label, ": unexpected flank_pain levels.")
  }
  x$flank_pain <- factor(pain, levels = c("No", "Yes"))

  outcome <- trimws(as.character(x$impacted_stone))
  outcome[outcome == "Yes"] <- "Impacted"
  outcome[outcome == "No"] <- "Non-impacted"
  if (!all(outcome %in% c("Impacted", "Non-impacted"))) {
    stop(label, ": unexpected outcome coding.")
  }
  x$impacted_stone <- factor(outcome,
                             levels = c("Impacted", "Non-impacted"))
  if (any(table(x$impacted_stone) < 5L)) {
    stop(label, " needs at least five events and five non-events.")
  }
  x
}

development <- prepare_cohort(read_cohort(development_file), "Development")
external <- prepare_cohort(read_cohort(external_file), "External validation")

# An exact match of all locked predictors plus outcome suggests reused patients.
row_signature <- function(x) {
  apply(x[c(predictors, "impacted_stone")], 1L, paste, collapse = "|")
}
if (any(row_signature(external) %in% row_signature(development))) {
  stop("External validation includes rows matching the development data.")
}

nested <- readRDS(nested_file)
if (!"all_model_patient_oof" %in% names(nested) ||
    !"all_model_summary" %in% names(nested) ||
    !"settings" %in% names(nested)) {
  stop("Nested results do not include RF out-of-fold predictions/metadata.")
}
oof <- as.data.frame(nested$all_model_patient_oof)
required_oof <- c("patient_id", "model", "truth", "probability",
                  "n_predictions")
if (!all(required_oof %in% names(oof))) {
  stop("Nested RF patient-level OOF columns are incomplete.")
}
oof <- oof[as.character(oof$model) == "random_forest", , drop = FALSE]
outer_repeats <- as.integer(nested$settings$value[
  nested$settings$setting == "outer_repeats"
])
rf_folds <- nested$all_model_summary$completed_outer_folds[
  nested$all_model_summary$model == "random_forest"
]
if (length(outer_repeats) != 1L || length(rf_folds) != 1L ||
    rf_folds != 25L || nrow(oof) != nrow(development) ||
    anyDuplicated(oof$patient_id) ||
    !all(oof$n_predictions == outer_repeats) ||
    !setequal(as.character(oof$patient_id),
              as.character(development$patient_id))) {
  stop("RF OOF data do not cover the 545 development patients across 25 folds.")
}
dev_match <- match(as.character(oof$patient_id),
                   as.character(development$patient_id))
if (!identical(as.character(oof$truth),
               as.character(development$impacted_stone[dev_match]))) {
  stop("OOF outcomes do not match the development data.")
}
test <- data.frame(
  cohort = "Development RF nested OOF",
  y = as.integer(as.character(oof$truth) == "Impacted"),
  prob = as.numeric(oof$probability)
)
if (any(!is.finite(test$prob) | test$prob < 0 | test$prob > 1)) {
  stop("RF out-of-fold probabilities are invalid.")
}

predict_locked <- function(x, cohort) {
  probabilities <- predict(bundle$fitted_workflow, new_data = x,
                           type = "prob")
  if (!".pred_Impacted" %in% names(probabilities)) {
    stop("The locked model did not return .pred_Impacted probabilities.")
  }
  p <- as.numeric(probabilities$.pred_Impacted)
  if (length(p) != nrow(x) || any(!is.finite(p) | p < 0 | p > 1)) {
    stop(cohort, ": invalid locked-model probabilities.")
  }
  data.frame(cohort = cohort, y = as.integer(x$impacted_stone == "Impacted"),
             prob = p)
}
ext <- predict_locked(external, "External validation")

# ---- 3. ROC, flexible calibration and net benefit -------------------------
roc_result <- function(d) {
  roc <- pROC::roc(factor(d$y, levels = c(0, 1)), d$prob,
                   levels = c("0", "1"), direction = "<", quiet = TRUE)
  ci <- as.numeric(pROC::ci.auc(roc, method = "delong"))
  list(curve = data.frame(fpr = 1 - roc$specificities,
                          tpr = roc$sensitivities),
       auc = as.numeric(pROC::auc(roc)), ci_low = ci[1], ci_high = ci[3])
}

calibration_result <- function(d, reps, rng_seed) {
  set.seed(rng_seed)
  eps <- 1e-6
  lp <- qlogis(pmin(1 - eps, pmax(eps, d$prob)))
  limits <- range(lp)
  knots <- as.numeric(quantile(lp, c(1 / 3, 2 / 3)))
  if (length(unique(lp)) < 10L || anyDuplicated(knots)) {
    stop("Too few unique prediction values for flexible calibration.")
  }
  grid <- seq(quantile(d$prob, 0.05), quantile(d$prob, 0.95),
              length.out = 101L)
  basis <- function(z) splines::ns(z, knots = knots,
                                   Boundary.knots = limits)
  design <- cbind(1, basis(lp))
  grid_design <- cbind(1, basis(qlogis(pmin(1 - eps, pmax(eps, grid)))))

  fit_curve <- function(indices) {
    fit <- suppressWarnings(stats::glm.fit(
      x = design[indices, , drop = FALSE], y = d$y[indices],
      family = stats::binomial()
    ))
    if (any(!is.finite(fit$coefficients))) return(rep(NA_real_, length(grid)))
    plogis(drop(grid_design %*% fit$coefficients))
  }
  estimate <- fit_curve(seq_len(nrow(d)))
  if (anyNA(estimate)) stop("Flexible calibration fit failed.")
  simulations <- replicate(reps, fit_curve(sample.int(nrow(d), replace = TRUE)))
  good <- colSums(is.finite(simulations)) == nrow(simulations)
  if (sum(good) < 0.8 * reps) stop("Too many bootstrap calibration fits failed.")
  bounds <- t(apply(simulations[, good, drop = FALSE], 1L,
                    quantile, probs = c(0.025, 0.975)))

  offset_fit <- suppressWarnings(glm(d$y ~ 1 + offset(lp),
                                     family = binomial()))
  slope_fit <- suppressWarnings(glm(d$y ~ lp, family = binomial()))
  list(curve = data.frame(predicted = grid, observed = estimate,
                          lower = bounds[, 1], upper = bounds[, 2]),
       intercept = unname(coef(offset_fit)[1]),
       slope = unname(coef(slope_fit)[2]),
       brier = mean((d$y - d$prob)^2), successful_bootstraps = sum(good))
}

decision_curve <- function(d, cutoffs) {
  n <- nrow(d)
  prevalence <- mean(d$y)
  do.call(rbind, lapply(cutoffs, function(pt) {
    positive <- d$prob >= pt
    data.frame(threshold = pt,
               strategy = c("RF", "Act on all", "Act on none"),
               net_benefit = c(
                 sum(positive & d$y == 1L) / n -
                   sum(positive & d$y == 0L) / n * pt / (1 - pt),
                 prevalence - (1 - prevalence) * pt / (1 - pt), 0
               ))
  }))
}

roc_test <- roc_result(test)
roc_ext <- roc_result(ext)
cal_test <- calibration_result(test, bootstrap_reps, seed)
cal_ext <- calibration_result(ext, bootstrap_reps, seed + 1L)
dca_test <- decision_curve(test, thresholds)
dca_ext <- decision_curve(ext, thresholds)
dca_range <- range(c(dca_test$net_benefit, dca_ext$net_benefit))

# ---- 4. Six panels, matched scales and journal-friendly style -------------
library(ggplot2)
ink <- "#243746"
blue <- "#286B91"
orange <- "#BD6944"
grey <- "#8B9298"
theme_pub <- theme_classic(base_size = 10, base_family = "sans") +
  theme(plot.title = element_text(face = "bold", size = 10.4,
                                  colour = ink, margin = margin(b = 3)),
        plot.subtitle = element_text(size = 8.5, colour = ink,
                                     margin = margin(b = 7)),
        plot.title.position = "plot",
        axis.title = element_text(size = 9.2, colour = ink),
        axis.text = element_text(size = 8.5, colour = ink),
        legend.position = "none", legend.title = element_blank(),
        legend.text = element_text(size = 8.2, colour = ink),
        legend.key.width = grid::unit(13, "pt"),
        legend.spacing.x = grid::unit(3, "pt"),
        plot.margin = margin(6, 9, 5, 7))

make_roc <- function(out, tag, cohort, source) {
  ggplot(out$curve, aes(fpr, tpr)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                linewidth = 0.45, colour = grey) +
    geom_path(linewidth = 0.95, colour = blue) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = paste(tag, paste0(cohort, ": ROC"), sep = "   "),
         subtitle = sprintf("%s  |  AUC %.3f (95%% CI %.3f–%.3f)",
                            source, out$auc, out$ci_low, out$ci_high),
         x = "1 - specificity", y = "Sensitivity") + theme_pub
}

make_calibration <- function(d, out, tag, cohort, source) {
  ggplot(out$curve, aes(predicted, observed)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                linewidth = 0.45, colour = grey) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = blue, alpha = 0.17) +
    geom_line(linewidth = 0.95, colour = blue) +
    geom_rug(data = d, aes(x = prob, colour = factor(y)),
             inherit.aes = FALSE, sides = "b", alpha = 0.55,
             length = grid::unit(0.028, "npc"), show.legend = FALSE) +
    scale_colour_manual(values = c(`0` = blue, `1` = orange)) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = paste(tag, paste0(cohort, ": calibration"), sep = "   "),
         subtitle = paste0(source, "  |  flexible fit and 95% CI"),
         x = "Predicted probability", y = "Observed proportion") + theme_pub
}

make_dca <- function(d, tag, cohort, source) {
  d$strategy <- factor(d$strategy,
                       levels = c("RF", "Act on all", "Act on none"))
  ggplot(d, aes(threshold, net_benefit, colour = strategy,
                linetype = strategy)) +
    geom_line(linewidth = 0.85) +
    scale_colour_manual(values = c("RF" = blue,
                                   "Act on all" = grey,
                                   "Act on none" = ink),
                        labels = c("RF", "Treat all", "Treat none")) +
    scale_linetype_manual(values = c("RF" = "solid",
                                      "Act on all" = "dashed",
                                      "Act on none" = "dotted"),
                          labels = c("RF", "Treat all", "Treat none")) +
    coord_cartesian(xlim = range(thresholds), ylim = dca_range,
                    expand = FALSE) +
    labs(title = paste(tag, paste0(cohort, ": decision curve"), sep = "   "),
         subtitle = paste0(source, "  |  thresholds 0.10–0.60"),
         x = "Threshold probability", y = "Net benefit") + theme_pub +
    theme(legend.position = "bottom", legend.direction = "horizontal",
          legend.justification = "center", legend.margin = margin(t = 1),
          legend.box.margin = margin(0, 0, 0, 0)) +
    guides(colour = guide_legend(nrow = 1),
           linetype = guide_legend(nrow = 1))
}

panels <- list(
  make_roc(roc_test, "A", "Development", "Nested OOF"),
  make_roc(roc_ext, "B", "External", "Locked RF"),
  make_calibration(test, cal_test, "C", "Development", "Nested OOF"),
  make_calibration(ext, cal_ext, "D", "External", "Locked RF"),
  make_dca(dca_test, "E", "Development", "Nested OOF"),
  make_dca(dca_ext, "F", "External", "Locked RF")
)
figure <- patchwork::wrap_plots(panels, ncol = 2, byrow = TRUE,
                                widths = c(1, 1), heights = c(1, 1, 1.07))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
pdf_file <- file.path(output_dir, "Figure3_RF_nested_OOF_external.pdf")
tiff_file <- file.path(output_dir, "Figure3_RF_nested_OOF_external.tiff")
png_file <- file.path(output_dir, "Figure3_RF_nested_OOF_external.png")
pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
ggsave(pdf_file, figure, width = 9, height = 11.2, units = "in",
       device = pdf_device, bg = "white")
ggsave(tiff_file, figure, width = 9, height = 11.2, units = "in",
       dpi = 600, compression = "lzw", bg = "white")
ggsave(png_file, figure, width = 9, height = 11.2, units = "in",
       dpi = 300, bg = "white")

audit <- data.frame(
  cohort = c("Development RF nested OOF", "External validation"),
  source = c(normalizePath(nested_file, winslash = "/"),
             normalizePath(external_file, winslash = "/")),
  n = c(nrow(test), nrow(ext)), events = c(sum(test$y), sum(ext$y)),
  auc = c(roc_test$auc, roc_ext$auc),
  auc_95_low = c(roc_test$ci_low, roc_ext$ci_low),
  auc_95_high = c(roc_test$ci_high, roc_ext$ci_high),
  brier = c(cal_test$brier, cal_ext$brier),
  calibration_intercept = c(cal_test$intercept, cal_ext$intercept),
  calibration_slope = c(cal_test$slope, cal_ext$slope),
  curve_bootstraps = c(cal_test$successful_bootstraps,
                       cal_ext$successful_bootstraps)
)
write.csv(audit, file.path(output_dir, "Figure3_numeric_audit.csv"),
          row.names = FALSE)
writeLines(c(
  "Figure 3. RF model discrimination, calibration, and decision curves.",
  "Panels A, C and E use development-cohort, patient-level out-of-fold",
  "probabilities from the RF candidate in nested cross-validation,",
  "averaged across outer repeats. They do not evaluate the final refitted RF.",
  "Panels B, D and F use unchanged probabilities from the locked final RF",
  "in the independent external validation cohort.",
  "ROC AUC 95% CIs use DeLong's method. Calibration shows a natural cubic",
  "spline logistic curve (two fixed internal knots) over the central 90%",
  "of predicted probabilities, with patient-level bootstrap 95% intervals",
  "conditional on the predictions. Orange rug marks indicate impacted",
  "stones; blue marks indicate non-impacted stones.",
  sprintf("Decision curves evaluate thresholds %.2f to %.2f against action",
          min(thresholds), max(thresholds)),
  "for all or no patients. The clinical action and relevant threshold",
  "range must be defined before claiming clinical utility."
), file.path(output_dir, "Figure3_legend_draft.txt"))
message("Saved: ", pdf_file, "\nSaved: ", tiff_file,
        "\nSaved: ", png_file)
