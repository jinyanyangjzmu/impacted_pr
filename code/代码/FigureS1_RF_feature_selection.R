# Figure S1 | Detailed feature selection for the locked random forest
#
# A: final LASSO 5-fold CV in the complete development cohort (n = 545).
# B: coefficient paths from the *same saved* cv.glmnet fit.
# C: selection frequencies from the 5 x 5 nested-CV outer training sets.
# D: descriptive Spearman correlations among selected continuous predictors.
# Full figure title/caption are exported as separate text. Compact color keys
# stay in panels B/C so the plotted colors can be decoded without guessing.
#
# The full-cohort LASSO is the final variable-selection step; panel C describes
# resampling stability. Neither panel is an independent performance estimate.
# The correlation display is descriptive and does not re-screen predictors.
# This script never refits LASSO/RF and never reads the external cohort.
#
# Run in R/RStudio (optional arguments: bundle, development RDS,
# output folder, fully nested results RDS):
# Rscript FigureS1_RF_feature_selection.R \
#   "C:/Users/Lenovo/Desktop/impacted_pr/output/final_selected_model/final_random_forest_model_bundle.rds" \
#   "C:/Users/Lenovo/Desktop/impacted_pr/data/impacted_clean.rds" \
#   "C:/Users/Lenovo/Desktop/impacted_pr/output/Figure_S1_feature_selection" \
#   "C:/Users/Lenovo/Desktop/impacted_pr/output/fully_nested_lasso_9models/fully_nested_lasso_9models_results.rds"

required_packages <- c("ggplot2", "patchwork", "scales", "glmnet")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Install R packages: ", paste(missing_packages, collapse = ", "))
}
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
project <- "C:/Users/Lenovo/Desktop/impacted_pr"
bundle_file <- if (length(args) >= 1L) args[[1L]] else file.path(
  project, "output", "final_selected_model", "final_random_forest_model_bundle.rds"
)
development_file <- if (length(args) >= 2L) args[[2L]] else file.path(
  project, "data", "impacted_clean.rds"
)
output_dir <- if (length(args) >= 3L) args[[3L]] else file.path(
  project, "output", "Figure_S1_feature_selection"
)
default_nested_file <- file.path(
  project, "output", "fully_nested_lasso_9models",
  "fully_nested_lasso_9models_results.rds"
)
if (length(args) > 4L) stop("Supply at most four arguments.")
if (!file.exists(bundle_file)) stop("Locked RF bundle missing: ", bundle_file)
if (!file.exists(development_file)) stop("Development data missing: ", development_file)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

bundle <- readRDS(bundle_file)
required_fields <- c(
  "selected_model", "selected_predictors", "selected_predictor_table",
  "lasso_cv_fit", "lasso_summary", "lasso_selected_terms", "configuration"
)
if (!all(required_fields %in% names(bundle))) {
  stop("Bundle missing: ", paste(setdiff(required_fields, names(bundle)), collapse = ", "))
}
if (!identical(bundle$selected_model, "random_forest")) {
  stop("Expected the finalized random_forest bundle.")
}
cv <- bundle$lasso_cv_fit
if (!inherits(cv, "cv.glmnet") || is.null(cv$glmnet.fit$beta)) {
  stop("Saved cv.glmnet coefficient path is missing.")
}
conf <- as.data.frame(bundle$configuration)
if (!all(c("setting", "value") %in% names(conf))) {
  stop("Malformed configuration in model bundle.")
}
config <- function(setting) {
  value <- as.character(conf$value[conf$setting == setting])
  if (length(value) == 1L) value else NA_character_
}
nested_file <- if (length(args) >= 4L) args[[4L]] else {
  saved_path <- config("nested_results_file")
  if (!is.na(saved_path) && file.exists(saved_path)) saved_path else default_nested_file
}
if (!file.exists(nested_file)) stop("Fully nested CV results missing: ", nested_file)
recorded_md5 <- config("data_md5")
if (is.na(recorded_md5) ||
    !identical(unname(tools::md5sum(development_file)), recorded_md5)) {
  stop("Development RDS checksum differs from the data used for the locked RF.")
}
recorded_nested_md5 <- config("nested_results_md5")
if (is.na(recorded_nested_md5) ||
    !identical(unname(tools::md5sum(nested_file)), recorded_nested_md5)) {
  stop("Nested results checksum differs from the results used to lock the RF.")
}

dat <- as.data.frame(readRDS(development_file), check.names = FALSE)
selected <- as.character(bundle$selected_predictors)
if (anyNA(selected) || !length(selected) || anyDuplicated(selected)) {
  stop("Invalid selected-predictor list.")
}
outcome <- if (is.list(bundle$outcome) && length(bundle$outcome$name) == 1L) {
  bundle$outcome$name
} else {
  "impacted_stone"
}
id_var <- "patient_id"
exclusions <- c(
  "stone_width", "hu_above_below_ratio", "alcohol_use",
  "hydronephrosis_grade", "ureteral_wall_area"
)
needed <- c(id_var, outcome, exclusions, selected)
if (!all(needed %in% names(dat))) {
  stop("Missing development columns: ", paste(setdiff(needed, names(dat)), collapse = ", "))
}
if (anyNA(dat)) stop("Development data have missing values; check input file.")
if (anyDuplicated(dat[[id_var]])) stop("Patient identifiers must be unique.")
factor_columns <- setdiff(
  names(dat)[vapply(dat, function(x) is.character(x) || is.logical(x), logical(1L))],
  id_var
)
dat[factor_columns] <- lapply(dat[factor_columns], factor)
candidate <- setdiff(names(dat), c(id_var, outcome, exclusions))
if (!all(selected %in% candidate)) stop("Selected predictors are not candidates.")

stability <- as.data.frame(bundle$selected_predictor_table)
stability_cols <- c(
  "predictor", "selected_outer_folds", "successful_lasso_folds",
  "selection_frequency"
)
if (!all(stability_cols %in% names(stability))) {
  stop("Outer-fold LASSO stability columns are missing from the bundle.")
}
if (!identical(as.character(stability$predictor), selected)) {
  stop("Saved selected-predictor table disagrees with final selected predictors.")
}
if (anyNA(stability[stability_cols]) ||
    any(as.numeric(stability$successful_lasso_folds) <= 0) ||
    any(abs(as.numeric(stability$selection_frequency) -
            as.numeric(stability$selected_outer_folds) /
            as.numeric(stability$successful_lasso_folds)) > 1e-8)) {
  stop("Invalid outer-fold selection frequencies in the model bundle.")
}

lambda_used <- as.numeric(bundle$lasso_summary$lambda_used[[1L]])
lambda_rule <- as.character(bundle$lasso_summary$lambda_rule[[1L]])
if (length(lambda_used) != 1L || !is.finite(lambda_used) || lambda_used <= 0 ||
    !lambda_rule %in% c("lambda.1se", "lambda.min_fallback")) {
  stop("Invalid final LASSO lambda or selection rule.")
}
if (!isTRUE(all.equal(lambda_used, as.numeric(cv[[
  if (lambda_rule == "lambda.1se") "lambda.1se" else "lambda.min"
]])))) stop("Saved LASSO rule and lambda disagree.")

label_map <- c(
  flank_pain = "Flank pain",
  stone_attenuation_hu = "Stone attenuation (HU)",
  proximal_ureter_width = "Proximal ureter width",
  ureteral_wall_thickness = "Ureteral wall thickness",
  stone_length = "Stone length",
  hydronephrosis_area = "Hydronephrosis area (APH)"
)
label_predictor <- function(x) {
  y <- unname(label_map[x])
  missing <- is.na(y)
  y[missing] <- tools::toTitleCase(gsub("_", " ", x[missing], fixed = TRUE))
  y
}
short_labels <- c(
  flank_pain = "Flank pain",
  stone_attenuation_hu = "Stone HU",
  proximal_ureter_width = "WUPC",
  ureteral_wall_thickness = "UWT",
  stone_length = "Stone length",
  hydronephrosis_area = "APH"
)
short_predictor <- function(x) {
  y <- unname(short_labels[x])
  y[is.na(y)] <- label_predictor(x[is.na(y)])
  y
}
ink <- "#22313D"
blue <- "#216C91"
red <- "#BB4D45"
muted <- "#AAB6BF"
font <- if (.Platform$OS.type == "windows") "Arial" else "sans"
theme_pub <- function() {
  theme_classic(base_family = font, base_size = 10.3) +
    theme(
      axis.title = element_text(size = 10, colour = ink),
      axis.text = element_text(size = 9, colour = ink),
      legend.position = "none",
      plot.margin = margin(11, 13, 9, 9)
    )
}

# A | Final LASSO cross-validation; error bars show SE, not confidence intervals.
cv_data <- data.frame(
  lambda = as.numeric(cv$lambda),
  cvm = as.numeric(cv$cvm),
  cvsd = as.numeric(cv$cvsd),
  nzero = as.integer(cv$nzero)
)
if (any(!is.finite(as.matrix(cv_data))) || any(cv_data$lambda <= 0)) {
  stop("Nonfinite saved LASSO CV values.")
}
cv_data$x <- log(cv_data$lambda)
lambda_min <- as.numeric(cv$lambda.min)
lambda_1se <- as.numeric(cv$lambda.1se)
if (any(!is.finite(c(lambda_min, lambda_1se))) ||
    any(c(lambda_min, lambda_1se) <= 0)) stop("Invalid CV lambda choices.")
pick <- which.min(abs(log(cv_data$lambda) - log(lambda_used)))
top_index <- unique(as.integer(round(seq(1, nrow(cv_data), length.out = 7L))))
panel_a <- ggplot(cv_data, aes(x, cvm)) +
  geom_linerange(aes(ymin = cvm - cvsd, ymax = cvm + cvsd),
                 colour = muted, linewidth = .20, alpha = .72) +
  geom_line(colour = ink, linewidth = .65) +
  geom_point(colour = ink, size = .65) +
  geom_vline(xintercept = log(lambda_min), colour = blue,
             linetype = "22", linewidth = .7) +
  geom_vline(xintercept = log(lambda_1se), colour = red,
             linetype = "dotted", linewidth = .7) +
  geom_point(data = cv_data[pick, , drop = FALSE], shape = 21,
             size = 2.9, stroke = .9, fill = "white", colour = red) +
  scale_x_continuous(
    sec.axis = dup_axis(
      breaks = cv_data$x[top_index],
      labels = cv_data$nzero[top_index],
      name = "Number of nonzero terms"
    )
  ) +
  labs(x = "log(lambda)", y = "Mean CV binomial deviance") +
  theme_pub() +
  theme(axis.title.x.top = element_text(size = 8.2, colour = "#53636F"),
        axis.text.x.top = element_text(size = 8.1))

# B | Recover glmnet's *saved* coefficient trajectories; retain original dummy
#     encoding so colored paths represent the final selected predictor terms.
model_formula <- stats::reformulate(candidate, response = outcome)
term_object <- stats::terms(model_formula, data = dat)
matrix_all <- stats::model.matrix(term_object, data = dat)
is_predictor <- colnames(matrix_all) != "(Intercept)"
term_labels <- attr(term_object, "term.labels")
dummy_map <- setNames(
  term_labels[attr(matrix_all, "assign")[is_predictor]],
  colnames(matrix_all)[is_predictor]
)
path_mat <- as.matrix(cv$glmnet.fit$beta)
path_lambda <- as.numeric(cv$glmnet.fit$lambda)
if (ncol(path_mat) != length(path_lambda) ||
    anyNA(dummy_map[rownames(path_mat)]) ||
    any(!is.finite(path_lambda)) || any(path_lambda <= 0)) {
  stop("Saved coefficient path cannot be mapped to the development design matrix.")
}
path_long <- data.frame(
  term = rep(rownames(path_mat), times = ncol(path_mat)),
  lambda = rep(path_lambda, each = nrow(path_mat)),
  coefficient = as.vector(path_mat),
  stringsAsFactors = FALSE
)
path_long$x <- log(path_long$lambda)
path_long$predictor <- unname(dummy_map[path_long$term])
final_terms <- as.character(bundle$lasso_selected_terms$term)
if (!length(final_terms) || anyNA(final_terms) ||
    !all(final_terms %in% unique(path_long$term))) {
  stop("Saved final LASSO terms cannot be identified in the coefficient path.")
}
path_long$final <- path_long$term %in% final_terms
path_long$display <- label_predictor(path_long$predictor)
# The reference LASSO plot expands the lambda range around lambda.min/1se.
# Use that same range while retaining the zero-coefficient endpoint.
path_xlim <- c(
  max(min(path_long$x), log(lambda_min) - .45),
  max(path_long$x)
)
if (diff(path_xlim) <= 0) stop("Invalid LASSO selection-range x limits.")
path_ticks <- which(cv_data$x >= path_xlim[1L] &
                    cv_data$x <= path_xlim[2L])
path_ticks <- unique(path_ticks[
  round(seq(1, length(path_ticks), length.out = min(6L, length(path_ticks))))
])
# A rare or low-variance nonretained dummy term can have a very large glmnet
# coefficient on the original predictor scale. Set the display range from the
# retained terms only; record every omitted value in the exported full path.
in_selection_range <- path_long$x >= path_xlim[1L] &
  path_long$x <= path_xlim[2L]
retained_values <- path_long$coefficient[
  path_long$final & in_selection_range
]
if (!length(retained_values) || any(!is.finite(retained_values))) {
  stop("The final LASSO paths contain no finite values in the selection range.")
}
retained_range <- range(c(0, retained_values))
padding <- max(.12 * diff(retained_range), .15)
path_ylim <- retained_range + c(-padding, padding)
clipped_other_terms <- unique(path_long$term[
  !path_long$final & in_selection_range &
    (path_long$coefficient < path_ylim[1L] |
       path_long$coefficient > path_ylim[2L])
])
predictor_colors <- c(
  flank_pain = "#CC6677",
  stone_attenuation_hu = "#A6761D",
  proximal_ureter_width = "#4D9221",
  ureteral_wall_thickness = "#009E73",
  stone_length = "#0072B2",
  hydronephrosis_area = "#9467BD"
)
unmapped <- setdiff(selected, names(predictor_colors))
if (length(unmapped)) predictor_colors <- c(
  predictor_colors,
  stats::setNames(grDevices::hcl.colors(length(unmapped), "Dark 3"), unmapped)
)
panel_b <- ggplot() +
  geom_line(data = path_long[!path_long$final, ],
            aes(x, coefficient, group = term), colour = muted,
            alpha = .62, linewidth = .36) +
  geom_line(data = path_long[path_long$final, ],
            aes(x, coefficient, group = term, colour = predictor),
            linewidth = .9, alpha = .92) +
  geom_hline(yintercept = 0, colour = "#B8C1C8", linewidth = .3) +
  geom_vline(xintercept = log(lambda_min), linetype = "22",
             colour = blue, linewidth = .7) +
  geom_vline(xintercept = log(lambda_1se), linetype = "dotted",
             colour = red, linewidth = .7) +
  scale_colour_manual(
    values = predictor_colors, breaks = selected,
    labels = short_predictor(selected), name = NULL
  ) +
  scale_x_continuous(
    sec.axis = dup_axis(
      breaks = cv_data$x[path_ticks],
      labels = cv_data$nzero[path_ticks],
      name = "Number of nonzero terms"
    )
  ) +
  labs(x = "log(lambda)", y = "Coefficient (log-odds scale)") +
  coord_cartesian(xlim = path_xlim, ylim = path_ylim) +
  guides(colour = guide_legend(nrow = 2, ncol = 3, byrow = TRUE,
                              override.aes = list(linewidth = 1.6))) +
  theme_pub() +
  theme(axis.title.x.top = element_text(size = 8.2, colour = "#53636F"),
        axis.text.x.top = element_text(size = 8.1),
        legend.position = "bottom",
        legend.justification = "left",
        legend.text = element_text(size = 7.5),
        legend.key.width = grid::unit(7, "mm"),
        legend.spacing.x = grid::unit(1.5, "mm"),
        legend.margin = margin(0, 0, 0, 0))

# C | All candidates if <=12, otherwise the final predictors plus the most
#     frequently selected other candidates. Full frequencies go to CSV.
#     These outer-fold frequencies come from the saved nested analysis.
nested <- readRDS(nested_file)
if (!"lasso_selection_frequency" %in% names(nested)) {
  stop("Nested CV results do not contain LASSO selection frequencies.")
}
nested_frequency <- as.data.frame(nested$lasso_selection_frequency)
if (!all(stability_cols %in% names(nested_frequency)) ||
    anyDuplicated(nested_frequency$predictor) ||
    !setequal(as.character(nested_frequency$predictor), candidate)) {
  stop("Nested selection-frequency data do not match candidate predictors.")
}
all_frequency <- nested_frequency[, stability_cols, drop = FALSE]
all_frequency$predictor <- as.character(all_frequency$predictor)
same <- match(selected, all_frequency$predictor)
if (any(abs(all_frequency$selection_frequency[same] -
            stability$selection_frequency) > 1e-8)) {
  stop("Bundle and nested results disagree on outer-fold frequencies.")
}
if (anyNA(all_frequency[stability_cols]) ||
    any(all_frequency$selected_outer_folds < 0) ||
    any(all_frequency$successful_lasso_folds <= 0) ||
    any(all_frequency$selection_frequency < 0 |
        all_frequency$selection_frequency > 1) ||
    any(abs(all_frequency$selection_frequency -
            all_frequency$selected_outer_folds /
            all_frequency$successful_lasso_folds) > 1e-8)) {
  stop("Invalid nested LASSO selection-frequency table.")
}
all_frequency$final <- all_frequency$predictor %in% selected
all_frequency$label <- label_predictor(all_frequency$predictor)
all_frequency <- all_frequency[
  order(-all_frequency$selection_frequency, all_frequency$predictor), ]
show_other <- head(all_frequency[!all_frequency$final, , drop = FALSE],
                   max(0L, 12L - length(selected)))
show_frequency <- rbind(all_frequency[all_frequency$final, , drop = FALSE],
                        show_other)
show_frequency <- show_frequency[
  order(show_frequency$selection_frequency, show_frequency$label), ]
show_frequency$label <- factor(show_frequency$label,
                               levels = unique(show_frequency$label))
show_frequency$counts <- sprintf(
  "%s/%s", show_frequency$selected_outer_folds,
  show_frequency$successful_lasso_folds
)
panel_c <- ggplot(show_frequency,
                  aes(x = selection_frequency, y = label, fill = final)) +
  geom_col(width = .68) +
  geom_text(aes(label = counts), hjust = -.12, size = 2.75,
            family = font, colour = ink) +
  scale_fill_manual(
    values = c("TRUE" = blue, "FALSE" = muted),
    breaks = c("TRUE", "FALSE"),
    labels = c("Final RF predictors", "Other candidates"),
    name = NULL
  ) +
  scale_x_continuous(limits = c(0, 1.16), breaks = seq(0, 1, .25),
                     labels = scales::label_percent(accuracy = 1),
                     expand = c(0, 0)) +
  labs(x = "LASSO selection frequency", y = NULL) +
  theme_pub() +
  theme(axis.text.y = element_text(size = 8),
        legend.position = "top",
        legend.justification = "left",
        legend.text = element_text(size = 8),
        legend.key.width = grid::unit(6, "mm"),
        legend.margin = margin(0, 0, 0, 0))

# D | Pairwise Spearman correlations of final CONTINUOUS predictors, within
#     the development cohort only. Binary/categorical predictors are omitted.
continuous <- selected[vapply(dat[selected], function(x) {
  is.numeric(x) && length(unique(x[!is.na(x)])) > 2L
}, logical(1L))]
if (length(continuous) < 2L) {
  stop("At least two selected continuous predictors are needed for panel D.")
}
rho <- stats::cor(dat[continuous], method = "spearman",
                  use = "pairwise.complete.obs")
if (any(!is.finite(rho))) stop("Undefined correlation among selected predictors.")
cor_data <- expand.grid(row = seq_along(continuous),
                        col = seq_along(continuous))
cor_data$rho <- rho[cbind(cor_data$row, cor_data$col)]
cor_data <- cor_data[cor_data$row >= cor_data$col, , drop = FALSE]
cor_data$row_label <- factor(label_predictor(continuous[cor_data$row]),
                             levels = rev(label_predictor(continuous)))
cor_data$col_label <- factor(short_predictor(continuous[cor_data$col]),
                             levels = short_predictor(continuous))
cor_data$number <- ifelse(cor_data$row == cor_data$col, "1.00",
                          sprintf("%.2f", cor_data$rho))
panel_d <- ggplot(cor_data, aes(col_label, row_label, fill = rho)) +
  geom_tile(colour = "white", linewidth = 1.6, width = .98, height = .98) +
  geom_text(aes(label = number, colour = abs(rho) >= .6),
            size = 3.05, family = font) +
  scale_colour_manual(values = c("FALSE" = ink, "TRUE" = "white"),
                      guide = "none") +
  scale_fill_gradient2(low = "#B96A5F", mid = "#F6F7F7",
                       high = "#2C809C", midpoint = 0,
                       limits = c(-1, 1), breaks = c(-1, -.5, 0, .5, 1),
                       guide = "none") +
  coord_fixed() +
  labs(x = NULL, y = NULL) +
  theme_pub() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1, vjust = 1,
                                   size = 8.2),
        axis.text.y = element_text(size = 8.2),
        axis.ticks = element_blank())

# Journal submission: store the complete figure title/caption in a separate
# editable text file. Keep only compact color keys inside the graphic.
fold_count <- unique(all_frequency$successful_lasso_folds)
if (length(fold_count) != 1L || fold_count <= 0) {
  stop("The number of successful nested LASSO folds is inconsistent.")
}
frequency_note <- if (nrow(show_frequency) == nrow(all_frequency)) {
  sprintf("All %d candidate predictors are displayed.", nrow(all_frequency))
} else {
  sprintf(
    "The plot displays the %d final predictors and %d other frequently selected candidates; frequencies for all %d candidates are provided in the accompanying CSV.",
    length(selected), nrow(show_other), nrow(all_frequency)
  )
}
crop_note <- if (length(clipped_other_terms)) {
  sprintf(
    "For readability, the y-axis is scaled to the retained terms; %d nonretained term path(s) extend beyond the displayed range. All coefficients are supplied in the accompanying CSV.",
    length(clipped_other_terms)
  )
} else {
  "The y-axis is scaled to the retained terms; all visible-range paths remain within the display limits."
}
figure_title <- "Figure S1. Detailed feature selection for the locked random-forest model."
figure_legend <- paste0(
  "(A) LASSO five-fold cross-validation in the complete development cohort ",
  "(n = ", nrow(dat), "). The points represent mean binomial deviance and ",
  "the grey vertical bars represent ±1 standard error (SE). The blue dashed ",
  "and red dotted lines mark lambda.min and lambda.1se, respectively. The ",
  "outlined point identifies the lambda used to select the final ",
  length(selected), " predictors (", lambda_rule, "). ",
  "(B) Coefficient paths from the saved final LASSO fit over the selection ",
  "range. Colored lines represent retained model-matrix terms, identified ",
  "by the compact color key below the panel; grey lines ",
  "represent other candidate terms. ", crop_note, " The upper axes in A and B ",
  "count nonzero model-matrix terms, which may differ from the number of ",
  "distinct clinical predictors. The LASSO coefficients describe screening ",
  "and should not be interpreted as effects of the final random forest. ",
  "(C) Frequency of LASSO selection across ", fold_count,
  " successful nested-CV outer training sets. Bars in blue identify ",
  "predictors retained by the final full-cohort LASSO; grey bars identify ",
  "other candidates. Labels show selected folds/successful folds. ",
  frequency_note,
  " (D) Descriptive pairwise Spearman rank correlations between final ",
  "continuous predictors in the development cohort. Numbers inside cells ",
  "are correlation coefficients; binary and categorical predictors were ",
  "excluded from this panel. HU = Hounsfield units; WUPC = proximal ",
  "ureteral width; UWT = ureteral wall thickness; APH = hydronephrosis ",
  "area. The correlation panel was not used to refit ",
  "the locked model."
)
legend_file <- file.path(output_dir, "Figure_S1_title_and_legend.txt")
writeLines(
  c(figure_title, "", paste(strwrap(figure_legend, width = 100),
                             collapse = "\n")),
  con = legend_file,
  useBytes = TRUE
)
figure <- ((panel_a | panel_b) / (panel_c | panel_d)) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.tag = element_text(face = "bold", size = 13, colour = ink,
                              family = font),
      plot.tag.position = c(.007, .99),
      plot.margin = margin(15, 16, 14, 16)
    )
  )

# Export vector PDF and 600-dpi TIFF + a 300-dpi PNG preview.
pdf_file <- file.path(output_dir, "Figure_S1_RF_feature_selection.pdf")
tiff_file <- file.path(output_dir, "Figure_S1_RF_feature_selection.tiff")
png_file <- file.path(output_dir, "Figure_S1_RF_feature_selection.png")
pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
ggsave(pdf_file, plot = figure, width = 9.0, height = 7.6,
       units = "in", device = pdf_device, bg = "white")
ggsave(tiff_file, plot = figure, width = 9.0, height = 7.6,
       units = "in", dpi = 600, compression = "lzw", bg = "white")
ggsave(png_file, plot = figure, width = 9.0, height = 7.6,
       units = "in", dpi = 300, bg = "white")

# Record plotted values and the complete stability table for review/reporting.
utils::write.csv(cv_data,
                 file.path(output_dir, "Figure_S1_LASSO_CV_values.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(all_frequency,
                 file.path(output_dir, "Figure_S1_outer_fold_frequency.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(as.data.frame(rho),
                 file.path(output_dir, "Figure_S1_Spearman_correlations.csv"),
                 row.names = TRUE, na = "")
utils::write.csv(as.data.frame(bundle$lasso_selected_terms),
                 file.path(output_dir, "Figure_S1_final_LASSO_terms.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(path_long,
                 file.path(output_dir, "Figure_S1_full_LASSO_coefficient_path.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(data.frame(
  clipped_term = clipped_other_terms,
  predictor = unname(dummy_map[clipped_other_terms])
), file.path(output_dir, "Figure_S1_paths_outside_display.csv"),
row.names = FALSE, na = "")
message("Figure S1 saved in: ", normalizePath(output_dir, winslash = "/"))
