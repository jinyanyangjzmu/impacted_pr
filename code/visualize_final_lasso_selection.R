# ============================================================
# Impacted stone study
# Publication-ready LASSO cross-validation and coefficient paths
#
# Reads the locked final-model bundle only. It does not refit LASSO,
# retune XGBoost, or read external-validation data.
#
# Usage:
# Rscript visualize_final_lasso_selection.R [model_bundle.rds] [output_dir]
# ============================================================

# 1. Setup ----------------------------------------------------
if (!requireNamespace("glmnet", quietly = TRUE)) {
  stop("Install the 'glmnet' package before running this script.")
}

args <- commandArgs(trailingOnly = TRUE)
default_dir <- "C:/Users/Lenovo/Desktop/impacted_pr/output/final_selected_model"
bundle_path <- if (length(args) >= 1L) {
  args[[1]]
} else {
  file.path(default_dir, "final_xgboost_model_bundle.rds")
}
output_dir <- if (length(args) >= 2L) args[[2]] else dirname(bundle_path)

if (!file.exists(bundle_path)) stop("Model bundle not found: ", bundle_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

bundle <- readRDS(bundle_path)
required_objects <- c(
  "lasso_cv_fit", "lasso_summary", "lasso_selected_terms",
  "selected_predictors"
)
missing_objects <- setdiff(required_objects, names(bundle))
if (length(missing_objects)) {
  stop("Missing bundle objects: ", paste(missing_objects, collapse = ", "))
}

cv_fit <- bundle$lasso_cv_fit
if (!inherits(cv_fit, "cv.glmnet") || is.null(cv_fit$glmnet.fit)) {
  stop("bundle$lasso_cv_fit is not a valid cv.glmnet object.")
}

glmnet_fit <- cv_fit$glmnet.fit
lambda_min <- as.numeric(cv_fit$lambda.min)
lambda_1se <- as.numeric(cv_fit$lambda.1se)
lambda_used <- as.numeric(bundle$lasso_summary$lambda_used[[1]])
lambda_rule <- as.character(bundle$lasso_summary$lambda_rule[[1]])

if (any(!is.finite(c(lambda_min, lambda_1se, lambda_used))) ||
    any(c(lambda_min, lambda_1se, lambda_used) <= 0)) {
  stop("The stored lambda values must be finite and positive.")
}

expected_lambda <- if (identical(lambda_rule, "lambda.1se")) {
  lambda_1se
} else if (identical(lambda_rule, "lambda.min_fallback")) {
  lambda_min
} else {
  stop("Unexpected stored lambda rule: ", lambda_rule)
}
if (!isTRUE(all.equal(lambda_used, expected_lambda, tolerance = 1e-10))) {
  stop("Stored lambda_used is inconsistent with the stored lambda rule.")
}

# 2. Extract plotting data ------------------------------------
# Use log(lambda) explicitly in both panels. This avoids ambiguity in
# glmnet plotting-method defaults and guarantees that the vertical lines
# use exactly the same x scale as the plotted data.
cv_x <- log(cv_fit$lambda)
cv_y <- cv_fit$cvm
cv_lower <- cv_fit$cvlo
cv_upper <- cv_fit$cvup
cv_nzero <- cv_fit$nzero

beta <- as.matrix(glmnet_fit$beta)
path_lambda <- as.numeric(glmnet_fit$lambda)
path_x <- log(path_lambda)

if (length(cv_x) != length(cv_y) ||
    length(cv_y) != length(cv_lower) ||
    length(cv_y) != length(cv_upper) ||
    ncol(beta) != length(path_lambda)) {
  stop("The stored cv.glmnet object has inconsistent dimensions.")
}

selected_terms <- intersect(
  as.character(bundle$lasso_selected_terms$term),
  rownames(beta)
)
if (!length(selected_terms)) {
  stop("No stored selected LASSO terms matched the coefficient matrix.")
}

# The complete path remains stored in the bundle. For the publication figure,
# show the range relevant to lambda selection and extend 1.5-fold toward weaker
# penalization. This prevents a remote, unstable weak-penalty coefficient from
# flattening all clinically relevant trajectories.
path_extension_factor <- 1.5
display_lower_lambda <- max(
  min(path_lambda),
  min(lambda_min, lambda_1se) / path_extension_factor
)
path_keep <- path_lambda >= display_lower_lambda

if (sum(path_keep) < 8L) {
  # Safe fallback when a saved path contains unusually few lambda values.
  nearest <- which.min(abs(log(path_lambda) - log(min(lambda_min, lambda_1se))))
  path_keep[seq_len(min(length(path_keep), nearest + 4L))] <- TRUE
}

beta_display <- beta[, path_keep, drop = FALSE]
x_display <- path_x[path_keep]

# Sparse top axes: glmnet's default labels every lambda and causes overlap.
sparse_index <- function(n, maximum = 7L) {
  unique(as.integer(round(seq(1, n, length.out = min(maximum, n)))))
}
cv_top <- sparse_index(length(cv_x))
path_top <- sparse_index(length(x_display))

# Highlight only terms retained at the locked lambda; all other paths remain
# visible in light gray. No endpoint numbers are printed, avoiding label piles.
selected_palette <- grDevices::hcl.colors(
  max(3L, length(selected_terms)), palette = "Dark 3"
)[seq_along(selected_terms)]
names(selected_palette) <- selected_terms

# 3. Plotting function ----------------------------------------
draw_lasso_figure <- function() {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)

  par(
    mfrow = c(1, 2),
    mar = c(4.8, 5.0, 5.6, 1.2),
    oma = c(0.3, 0.2, 0.2, 0.2),
    family = "sans",
    las = 1,
    mgp = c(3.0, 0.85, 0),
    tcl = -0.25,
    xaxs = "r",
    yaxs = "r"
  )

  # Panel A: mean cross-validated deviance +/- 1 standard error.
  y_range <- range(c(cv_lower, cv_upper), finite = TRUE)
  y_pad <- diff(y_range) * 0.04
  if (!is.finite(y_pad) || y_pad == 0) y_pad <- 0.05

  plot(
    cv_x, cv_y,
    type = "n",
    ylim = y_range + c(-y_pad, y_pad),
    xlab = expression(log(lambda)),
    ylab = "Mean cross-validated binomial deviance",
    xaxt = "n",
    main = ""
  )
  axis(1)
  segments(cv_x, cv_lower, cv_x, cv_upper, col = "#B8B8B8", lwd = 0.8)
  points(cv_x, cv_y, pch = 16, cex = 0.62, col = "#C44E52")
  abline(v = log(lambda_min), col = "#3B7EA1", lty = 2, lwd = 1.4)
  abline(v = log(lambda_1se), col = "#C44E52", lty = 3, lwd = 1.4)

  axis(
    3,
    at = cv_x[cv_top],
    labels = cv_nzero[cv_top],
    tick = FALSE,
    line = 0.15,
    cex.axis = 0.72
  )
  mtext("Number of nonzero coefficients", side = 3, line = 1.45, cex = 0.72)
  mtext("A", side = 3, line = 3.65, adj = 0, font = 2, cex = 1.15)
  mtext(
    "Cross-validation curve",
    side = 3, line = 3.65, adj = 0.09, font = 2, cex = 1.02
  )
  legend(
    "topright",
    legend = c(expression(lambda[min]), expression(lambda[1*SE])),
    col = c("#3B7EA1", "#C44E52"),
    lty = c(2, 3),
    lwd = 1.4,
    bty = "n",
    cex = 0.78,
    inset = 0.01
  )

  # Panel B: paths in the lambda range relevant to final selection.
  coefficient_range <- range(c(0, beta_display), finite = TRUE)
  coefficient_pad <- diff(coefficient_range) * 0.06
  if (!is.finite(coefficient_pad) || coefficient_pad == 0) coefficient_pad <- 0.1

  matplot(
    x_display,
    t(beta_display),
    type = "l",
    lty = 1,
    lwd = 0.65,
    col = "#D4D4D4",
    ylim = coefficient_range + c(-coefficient_pad, coefficient_pad),
    xlab = expression(log(lambda)),
    ylab = "Coefficient",
    xaxt = "n",
    main = ""
  )
  axis(1)

  for (term in selected_terms) {
    lines(
      x_display,
      beta_display[term, ],
      col = selected_palette[[term]],
      lwd = 1.55
    )
  }
  abline(h = 0, col = "#8A8A8A", lty = 3, lwd = 0.7)
  abline(v = log(lambda_min), col = "#3B7EA1", lty = 2, lwd = 1.4)
  abline(v = log(lambda_1se), col = "#C44E52", lty = 3, lwd = 1.4)

  path_nzero <- colSums(beta_display != 0)
  axis(
    3,
    at = x_display[path_top],
    labels = path_nzero[path_top],
    tick = FALSE,
    line = 0.15,
    cex.axis = 0.72
  )
  mtext("Number of nonzero coefficients", side = 3, line = 1.45, cex = 0.72)
  mtext("B", side = 3, line = 3.65, adj = 0, font = 2, cex = 1.15)
  mtext(
    "Coefficient paths (selection range)",
    side = 3, line = 3.65, adj = 0.09, font = 2, cex = 1.02
  )
}

# 4. Save PDF, PNG, and exact plotting metadata ---------------
pdf_path <- file.path(output_dir, "Figure_final_LASSO_selection.pdf")
png_path <- file.path(output_dir, "Figure_final_LASSO_selection.png")

pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
pdf_device(pdf_path, width = 11.8, height = 5.8, family = "sans", pointsize = 10)
draw_lasso_figure()
dev.off()

grDevices::png(
  png_path,
  width = 11.8,
  height = 5.8,
  units = "in",
  res = 600,
  bg = "white",
  pointsize = 10
)
draw_lasso_figure()
dev.off()

coef_at_lambda <- as.matrix(stats::coef(cv_fit, s = lambda_used))
coefficient_key <- data.frame(
  curve_id = seq_len(nrow(beta)),
  encoded_term = rownames(beta),
  selected_at_locked_lambda = rownames(beta) %in% selected_terms,
  coefficient_at_locked_lambda = coef_at_lambda[rownames(beta), 1],
  stringsAsFactors = FALSE
)
write.csv(
  coefficient_key,
  file.path(output_dir, "31_final_lasso_coefficient_key.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

plot_manifest <- data.frame(
  item = c(
    "lambda_rule", "lambda_used", "lambda_min", "lambda_1se",
    "selected_clinical_predictors", "selected_encoded_terms",
    "panel_B_minimum_lambda", "panel_B_path_extension_factor",
    "model_refitted_by_this_script"
  ),
  value = as.character(c(
    lambda_rule, lambda_used, lambda_min, lambda_1se,
    length(bundle$selected_predictors), length(selected_terms),
    min(path_lambda[path_keep]), path_extension_factor, "No"
  )),
  stringsAsFactors = FALSE
)
write.csv(
  plot_manifest,
  file.path(output_dir, "32_final_lasso_plot_manifest.csv"),
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

message(
  "LASSO figures created from the locked bundle without refitting.\n",
  "PDF: ", pdf_path, "\n",
  "PNG: ", png_path, "\n",
  "Panel B displays lambda >= ", signif(min(path_lambda[path_keep]), 4),
  "; the complete path remains stored in the bundle."
)
