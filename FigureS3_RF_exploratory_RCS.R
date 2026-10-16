# Figure S3 | Exploratory restricted cubic splines (RCS), impacted stone study
#
# The locked RF has six predictors: five continuous plus flank_pain (categorical).
# Plot the five continuous predictors only. For each predictor fit a separate
# logistic regression with a four-knot RCS for that predictor, adjusting for
# the other four continuous RF predictors as linear terms and flank_pain as a
# factor. Report odds ratios relative to that predictor's development-cohort
# median. This association analysis DOES NOT explain the fitted RF and its
# post-selection P-values are exploratory; no breakpoint search is performed.
#
# Rscript FigureS3_RF_exploratory_RCS.R [locked_RF_bundle.rds] \
#   [development_data.rds] [output_directory]
# Default output: C:/Users/Lenovo/Desktop/impacted_pr/output/Figure_S3_RCS
# Produces a vector PDF, 600-dpi TIFF, PNG preview, numeric CSVs, and a
# separate editable figure-title-and-legend text file.
# The example layout is used for visual styling only: this figure uses the
# impacted-stone development data and its own selected predictor measurements.

required <- c("ggplot2", "patchwork", "scales")
absent <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(absent)) stop("Install packages: ", paste(absent, collapse = ", "))
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 3L) stop("Supply at most three command-line arguments.")
project <- "C:/Users/Lenovo/Desktop/impacted_pr"
bundle_path <- if (length(args) >= 1L) args[[1L]] else file.path(
  project, "output", "final_selected_model", "final_random_forest_model_bundle.rds"
)
data_path <- if (length(args) >= 2L) args[[2L]] else file.path(
  project, "data", "impacted_clean.rds"
)
out_dir <- if (length(args) >= 3L) args[[3L]] else file.path(
  project, "output", "Figure_S3_RCS"
)
if (!file.exists(bundle_path)) stop("Locked RF bundle missing: ", bundle_path)
if (!file.exists(data_path)) stop("Development RDS missing: ", data_path)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

bundle <- readRDS(bundle_path)
if (!identical(bundle$selected_model, "random_forest") ||
    is.null(bundle$selected_predictors) || is.null(bundle$configuration) ||
    is.null(bundle$outcome)) {
  stop("Expected the locked RF model bundle with provenance and outcome labels.")
}
selected <- as.character(bundle$selected_predictors)
if (length(selected) != 6L || anyNA(selected) || anyDuplicated(selected) ||
    any(!grepl("^[[:alpha:]][[:alnum:]_.]*$", selected))) {
  stop("Expected six uniquely named selected RF predictors.")
}
configuration <- as.data.frame(bundle$configuration)
if (!all(c("setting", "value") %in% names(configuration))) {
  stop("Malformed model-bundle configuration.")
}
recorded_md5 <- configuration$value[configuration$setting == "data_md5"]
if (length(recorded_md5) != 1L ||
    !identical(as.character(recorded_md5), unname(tools::md5sum(data_path)))) {
  stop("The development RDS checksum differs from the data used to lock the RF.")
}

data <- as.data.frame(readRDS(data_path), check.names = FALSE)
outcome_name <- as.character(bundle$outcome$name)
positive <- as.character(bundle$outcome$positive)
negative <- as.character(bundle$outcome$negative)
if (length(outcome_name) != 1L || length(positive) != 1L ||
    length(negative) != 1L || !all(c(selected, outcome_name) %in% names(data))) {
  stop("Model predictors or recorded outcome are missing from development data.")
}
model_data <- data[, c(selected, outcome_name), drop = FALSE]
if (anyNA(model_data)) stop("The development model data contain missing values.")
truth <- trimws(as.character(model_data[[outcome_name]]))
if (!setequal(unique(truth), c(positive, negative))) {
  stop("The outcome coding does not match the locked bundle.")
}
model_data$outcome01 <- as.integer(truth == positive)
model_data[[outcome_name]] <- NULL

# Only genuinely continuous numeric measurements receive spline terms.
# Binary, categorical, and sparsely valued measurements are not treated as RCS.
is_continuous <- vapply(model_data[selected], function(x) {
  is.numeric(x) && all(is.finite(x)) && length(unique(x)) >= 10L
}, logical(1L))
continuous <- selected[is_continuous]
categorical <- setdiff(selected, continuous)
if (length(continuous) != 5L ||
    !identical(categorical, "flank_pain")) {
  stop("Expected five continuous RF predictors and categorical flank_pain; ",
       "inspect bundle and input-variable types before running Figure S3.")
}
model_data[categorical] <- lapply(model_data[categorical], function(x) {
  factor(trimws(as.character(x)))
})
if (any(vapply(model_data[categorical], nlevels, integer(1L)) < 2L)) {
  stop("The categorical adjustment predictor has fewer than two levels.")
}

# Present UWT first, followed by the other locked imaging predictors.
preferred_order <- c(
  "ureteral_wall_thickness", "stone_attenuation_hu",
  "proximal_ureter_width", "stone_length", "hydronephrosis_area"
)
continuous <- c(intersect(preferred_order, continuous),
                setdiff(continuous, preferred_order))
display_names <- c(
  ureteral_wall_thickness = "Ureteral wall thickness",
  stone_attenuation_hu = "Stone attenuation (HU)",
  proximal_ureter_width = "Proximal ureter width (WUPC)",
  stone_length = "Stone length",
  hydronephrosis_area = "Hydronephrosis area (APH)"
)
label_for <- function(x) {
  label <- unname(display_names[x])
  if (is.na(label)) tools::toTitleCase(gsub("_", " ", x)) else label
}
ink <- "#25343D"
curve_col <- "#D8785F"
ribbon_col <- "#F7D6C9"
hist_col <- "#C7E0E6"
hist_border <- "#7E9DA5"
font <- if (.Platform$OS.type == "windows") "Arial" else "sans"

format_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < .001) "<0.001" else sprintf("%.3f", p)
}

# Explicit knot values ensure all fits, model-matrix contrasts, and plots use
# exactly the same natural cubic (restricted cubic) spline basis.
fit_exploratory_spline <- function(variable) {
  values <- model_data[[variable]]
  knots <- as.numeric(stats::quantile(
    values, probs = c(.05, .35, .65, .95), names = FALSE, type = 7
  ))
  if (any(!is.finite(knots)) || any(diff(knots) <= 0)) {
    stop("Non-distinct 5/35/65/95 percentile knots for: ", variable)
  }
  fmt <- function(z) format(z, digits = 17L, scientific = TRUE, trim = TRUE)
  spline_term <- sprintf(
    "splines::ns(%s, knots = c(%s, %s), Boundary.knots = c(%s, %s))",
    variable, fmt(knots[2L]), fmt(knots[3L]),
    fmt(knots[1L]), fmt(knots[4L])
  )
  adjusters <- setdiff(selected, variable)
  spline_formula <- stats::reformulate(c(spline_term, adjusters),
                                      response = "outcome01")
  linear_formula <- stats::reformulate(c(variable, adjusters),
                                      response = "outcome01")
  no_exposure_formula <- stats::reformulate(adjusters,
                                           response = "outcome01")
  fit <- stats::glm(spline_formula, data = model_data,
                    family = stats::binomial())
  fit_linear <- stats::glm(linear_formula, data = model_data,
                           family = stats::binomial())
  fit_none <- stats::glm(no_exposure_formula, data = model_data,
                         family = stats::binomial())
  if (!all(vapply(list(fit, fit_linear, fit_none),
                  function(z) isTRUE(z$converged) &&
                    !anyNA(stats::coef(z)), logical(1L)))) {
    stop("A logistic model did not converge or has aliased coefficients: ",
         variable)
  }

  lr_p <- function(smaller, larger) {
    df <- stats::df.residual(smaller) - stats::df.residual(larger)
    chi2 <- stats::deviance(smaller) - stats::deviance(larger)
    if (df < 1L || !is.finite(chi2) || chi2 < -1e-7) {
      stop("Invalid nested likelihood-ratio comparison for: ", variable)
    }
    list(chisq = max(0, chi2), df = df,
         p = stats::pchisq(max(0, chi2), df = df, lower.tail = FALSE))
  }
  overall <- lr_p(fit_none, fit)
  nonlinear <- lr_p(fit_linear, fit)

  # Center the odds ratio exactly at the predictor's median. Other variables
  # are fixed at their medians or modal levels; they cancel in the contrast
  # because this model has no interaction terms.
  reference <- stats::median(values)
  xgrid <- sort(unique(c(seq(knots[1L], knots[4L], length.out = 251L),
                         reference)))
  typical <- lapply(model_data[selected], function(x) {
    if (is.factor(x)) {
      frequencies <- table(x)
      names(frequencies)[which.max(frequencies)]
    } else {
      stats::median(x)
    }
  })
  new_data <- model_data[rep(1L, length(xgrid)), selected, drop = FALSE]
  for (name in selected) {
    if (is.factor(model_data[[name]])) {
      new_data[[name]] <- factor(rep(typical[[name]], length(xgrid)),
                                 levels = levels(model_data[[name]]))
    } else {
      new_data[[name]] <- rep(typical[[name]], length(xgrid))
    }
  }
  new_data[[variable]] <- xgrid
  ref_data <- new_data[1L, , drop = FALSE]
  ref_data[[variable]] <- reference

  design <- stats::delete.response(stats::terms(fit))
  mm <- stats::model.matrix(design, data = new_data,
                            contrasts.arg = fit$contrasts, xlev = fit$xlevels)
  mm_ref <- stats::model.matrix(design, data = ref_data,
                                contrasts.arg = fit$contrasts, xlev = fit$xlevels)
  beta <- stats::coef(fit)
  if (!setequal(colnames(mm), names(beta)) ||
      !setequal(colnames(mm_ref), names(beta))) {
    stop("Model-matrix terms cannot be aligned to spline coefficients: ",
         variable)
  }
  delta <- sweep(mm[, names(beta), drop = FALSE], 2L,
                 mm_ref[1L, names(beta)], "-")
  vcov <- stats::vcov(fit)[names(beta), names(beta), drop = FALSE]
  log_or <- as.vector(delta %*% beta)
  se <- sqrt(pmax(0, rowSums((delta %*% vcov) * delta)))
  z <- stats::qnorm(.975)
  curve <- data.frame(
    predictor = variable, x = xgrid,
    odds_ratio = exp(log_or), lower = exp(log_or - z * se),
    upper = exp(log_or + z * se), reference = reference
  )
  if (any(!is.finite(as.matrix(curve[c("x", "odds_ratio", "lower", "upper")]))) ||
      any(curve$lower <= 0) || any(curve$upper <= 0)) {
    stop("Non-finite odds ratio or confidence interval for: ", variable)
  }
  info <- data.frame(
    predictor = variable, display_name = label_for(variable),
    n = nrow(model_data), events = sum(model_data$outcome01),
    reference_median = reference,
    knot_05 = knots[1L], knot_35 = knots[2L],
    knot_65 = knots[3L], knot_95 = knots[4L],
    overall_lr_chisq = overall$chisq, overall_df = overall$df,
    p_overall = overall$p,
    nonlinear_lr_chisq = nonlinear$chisq,
    nonlinear_df = nonlinear$df, p_nonlinear = nonlinear$p
  )

  observations <- data.frame(x = values[
    values >= knots[1L] & values <= knots[4L]
  ])
  p_annotation <- paste0("Overall P = ", format_p(overall$p),
                         "\nNonlinear P = ", format_p(nonlinear$p))
  base_theme <- theme_classic(base_size = 10.4, base_family = font) +
    theme(
      axis.title = element_text(size = 9.2, colour = ink),
      axis.text = element_text(size = 8.2, colour = ink),
      axis.line = element_line(colour = "#64747B", linewidth = .35),
      axis.ticks = element_line(colour = "#64747B", linewidth = .35)
    )

  spline_panel <- ggplot(curve, aes(x = x, y = odds_ratio)) +
    geom_ribbon(aes(ymin = lower, ymax = upper),
                fill = ribbon_col, alpha = .72) +
    geom_hline(yintercept = 1, linetype = "dashed",
               linewidth = .44, colour = "#818B90") +
    geom_vline(xintercept = reference, linetype = "dotted",
               linewidth = .42, colour = "#768187") +
    geom_line(colour = curve_col, linewidth = .93) +
    annotate("label", x = Inf, y = Inf, label = p_annotation,
             hjust = 1.04, vjust = 1.12, size = 2.75,
             lineheight = 1.20, colour = ink, fill = "white",
             alpha = .92, linewidth = 0, label.padding = grid::unit(.7, "mm")) +
    scale_y_log10(labels = scales::label_number(accuracy = .01)) +
    scale_x_continuous(breaks = scales::breaks_pretty(n = 4L)) +
    coord_cartesian(xlim = knots[c(1L, 4L)], clip = "on") +
    labs(x = NULL, y = "Adjusted odds ratio",
         tag = LETTERS[match(variable, continuous)]) +
    base_theme +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.line.x = element_blank(),
      axis.title.y = element_text(margin = margin(r = 4)),
      plot.tag = element_text(face = "bold", size = 13,
                              colour = ink, family = font),
      plot.tag.position = c(.015, .99),
      plot.margin = margin(11, 10, 0, 9)
    )

  # Keep the predictor distribution on its own count axis. Superimposing
  # counts on the odds-ratio axis would give the blue bars an arbitrary scale.
  count_panel <- ggplot(observations, aes(x = x)) +
    geom_histogram(bins = 25L, fill = hist_col, colour = hist_border,
                   linewidth = .28, boundary = knots[1L]) +
    scale_x_continuous(breaks = scales::breaks_pretty(n = 4L)) +
    scale_y_continuous(breaks = scales::breaks_pretty(n = 2L),
                       expand = expansion(mult = c(0, .08))) +
    coord_cartesian(xlim = knots[c(1L, 4L)], clip = "on") +
    labs(x = label_for(variable), y = "Count") +
    base_theme +
    theme(
      axis.title.x = element_text(size = 9.2, margin = margin(t = 5)),
      axis.title.y = element_text(size = 8, margin = margin(r = 4)),
      axis.text.y = element_text(size = 7.3),
      plot.margin = margin(1, 10, 11, 9)
    )
  panel <- patchwork::wrap_plots(spline_panel, count_panel,
                                 ncol = 1L, heights = c(3.25, 1))
  list(plot = panel, curve = curve, statistics = info)
}

results <- lapply(continuous, fit_exploratory_spline)
names(results) <- continuous
panels <- lapply(results, `[[`, "plot")
curves <- do.call(rbind, lapply(results, `[[`, "curve"))
statistics <- do.call(rbind, lapply(results, `[[`, "statistics"))
row.names(curves) <- row.names(statistics) <- NULL

# Three upper panels and two centered lower panels. In patchwork text designs,
# '#' marks an empty grid cell; '.' is treated as a named plot area.
design <- "AABBCC\n#DDEE#"
figure <- patchwork::wrap_plots(unname(panels), design = design) +
  patchwork::plot_annotation(
    theme = ggplot2::theme(
      plot.margin = ggplot2::margin(20, 18, 17, 18)
    )
  )

# Keep the submission graphic free of long panel titles or figure captions.
pdf_path <- file.path(out_dir, "Figure_S3_RF_exploratory_RCS.pdf")
tiff_path <- file.path(out_dir, "Figure_S3_RF_exploratory_RCS.tiff")
png_path <- file.path(out_dir, "Figure_S3_RF_exploratory_RCS.png")
pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
ggplot2::ggsave(pdf_path, figure, width = 12.6, height = 8.7,
                units = "in", device = pdf_device, bg = "white")
ggplot2::ggsave(tiff_path, figure, width = 12.6, height = 8.7,
                units = "in", dpi = 600, compression = "lzw", bg = "white")
ggplot2::ggsave(png_path, figure, width = 12.6, height = 8.7,
                units = "in", dpi = 300, bg = "white")

utils::write.csv(statistics,
                 file.path(out_dir, "Figure_S3_RCS_tests_and_knots.csv"),
                 row.names = FALSE)
utils::write.csv(curves,
                 file.path(out_dir, "Figure_S3_RCS_curve_points.csv"),
                 row.names = FALSE)

panel_details <- vapply(seq_along(continuous), function(i) {
  row <- statistics[i, ]
  sprintf("(%s) %s: P overall = %s; P nonlinearity = %s.",
          LETTERS[i], row$display_name,
          format_p(row$p_overall), format_p(row$p_nonlinear))
}, character(1L))
legend_text <- paste0(
  "Figure S3. Exploratory restricted cubic spline analyses of the continuous ",
  "predictors selected for the final random-forest model.\n\n",
  paste(panel_details, collapse = "\n"), "\n\n",
  "Analyses used the complete development cohort (n = ", nrow(model_data),
  "; impacted cases = ", sum(model_data$outcome01),
  "). Flank pain is categorical and was included as an adjustment factor, ",
  "not as a spline panel. Each panel comes from a separate logistic model ",
  "with four prespecified RCS knots at that predictor's 5th, 35th, 65th, ",
  "and 95th percentiles; the other selected continuous predictors were ",
  "entered linearly and flank pain categorically. Solid lines are adjusted ",
  "odds ratios for impacted stone relative to the predictor's median ",
  "(vertical dotted line); shading shows pointwise 95% confidence intervals. ",
  "Horizontal dashed lines show an odds ratio of 1. The pale blue histograms ",
  "below the curves show the observed number of patients at predictor values ",
  "within the plotted 5th-95th percentile range; they have a separate count ",
  "axis. Overall ",
  "and nonlinearity P values derive from likelihood-ratio comparisons of ",
  "nested logistic models (spline versus no exposure and spline versus ",
  "linear exposure, respectively). Each panel uses its own log-scaled ",
  "y-axis. These are post-selection exploratory associations, not ",
  "causal effects, independent contributions, validated cutpoints, or ",
  "effect curves of the locked random forest."
)
writeLines(legend_text,
           file.path(out_dir, "Figure_S3_title_and_legend.txt"),
           useBytes = TRUE)
message("Figure S3 saved to: ", normalizePath(out_dir, winslash = "/"))
