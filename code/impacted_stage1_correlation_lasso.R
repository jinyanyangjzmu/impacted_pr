# ============================================================
# Impacted stone study: correlation analysis and LASSO-CV
# Data: impacted_clean.rds (n = 545)
# Outcome: impacted_stone; positive class = Impacted
# This script does not run the nine-model comparison.
# For leakage-free model comparison, repeat this screening inside each
# outer-CV analysis fold; do not fix this whole-cohort list before outer CV.
# ============================================================

# 1. Packages and settings ------------------------------------
pkgs <- c("glmnet", "dplyr", "tidyr", "purrr", "ggplot2", "readr")
new_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)

suppressPackageStartupMessages({
  library(glmnet)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(readr)
})

set.seed(2026)

data_path <- "C:\\Users\\Lenovo\\Desktop\\impacted_pr\\data\\impacted_clean.rds"
if (!file.exists(data_path)) data_path <- file.choose()

out_dir <- file.path(dirname(data_path), "stage1_correlation_lasso")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

outcome <- "impacted_stone"
id_var <- "patient_id"
negative <- "Non-impacted"
positive <- "Impacted"
rho_cutoff <- 0.80
nfolds <- 5
n_repeats <- 20
plot_family <- "sans"  # Portable; maps to a system sans-serif font.

# Add only prespecified clinical/redundancy exclusions here.
# Do not exclude predictors according to univariable P values.
manual_drop <- character(0)

save_plot <- function(plot, name, width, height) {
  ggsave(file.path(out_dir, paste0(name, ".pdf")), plot,
         width = width, height = height, device = cairo_pdf)
  ggsave(file.path(out_dir, paste0(name, ".png")), plot,
         width = width, height = height, dpi = 600, bg = "white")
}

# 2. Data preparation -----------------------------------------
dat <- readRDS(data_path) |> as.data.frame(check.names = FALSE)
stopifnot(all(c(id_var, outcome) %in% names(dat)), !anyNA(dat))

dat[[outcome]] <- factor(trimws(as.character(dat[[outcome]])),
                         levels = c(negative, positive))
if (anyNA(dat[[outcome]])) stop("Unexpected values in impacted_stone.")

char_vars <- names(dat)[vapply(dat, is.character, logical(1))]
dat[char_vars] <- lapply(dat[char_vars], factor)

predictors <- setdiff(names(dat), c(id_var, outcome, manual_drop))
numeric_vars <- predictors[vapply(dat[predictors], is.numeric, logical(1))]
factor_vars <- predictors[vapply(dat[predictors], is.factor, logical(1))]
y <- as.integer(dat[[outcome]] == positive)

audit <- tibble(
  item = c("patients", "candidate_predictors", "impacted",
           "non_impacted", "missing_cells"),
  value = c(nrow(dat), length(predictors), sum(y == 1),
            sum(y == 0), sum(is.na(dat)))
)
write_excel_csv(audit, file.path(out_dir, "00_data_audit.csv"))

# 3. Correlation analysis -------------------------------------
# Continuous predictors: Spearman rho.
# Categorical predictors: Cramer's V. They are not converted to arbitrary
# integer codes for the Spearman analysis.

rho <- cor(dat[numeric_vars], method = "spearman")
write_excel_csv(
  data.frame(variable = rownames(rho), rho, check.names = FALSE),
  file.path(out_dir, "01_spearman_matrix.csv")
)

rho_pairs <- combn(numeric_vars, 2, simplify = FALSE) |>
  map_dfr(\(z) {
    rho_value <- rho[z[1], z[2]]
    tibble(
      variable_1 = z[1], variable_2 = z[2],
      rho = rho_value, abs_rho = abs(rho_value)
    )
  }) |>
  arrange(desc(abs_rho))

write_excel_csv(rho_pairs, file.path(out_dir, "02_all_spearman_pairs.csv"))
write_excel_csv(filter(rho_pairs, abs_rho >= rho_cutoff),
                file.path(out_dir, "03_high_correlation_pairs.csv"))

ord <- colnames(rho)[hclust(as.dist(1 - abs(rho)))$order]
rho_long <- as.data.frame(as.table(rho)) |>
  transmute(
    x = factor(Var1, levels = ord),
    y = factor(Var2, levels = rev(ord)),
    rho = Freq,
    mark = ifelse(abs(rho) >= rho_cutoff & abs(rho) < 1, "*", "")
  )

p_rho <- ggplot(rho_long, aes(x, y, fill = rho)) +
  geom_tile(color = "white", linewidth = 0.15) +
  geom_text(aes(label = mark), size = 4) +
  scale_fill_gradient2(low = "#2C7BB6", mid = "white", high = "#D7191C",
                       midpoint = 0, limits = c(-1, 1), name = "Spearman rho") +
  coord_fixed() +
  labs(title = "Spearman correlation among continuous predictors",
       subtitle = paste0("* |rho| >= ", rho_cutoff), x = NULL, y = NULL) +
  theme_minimal(base_size = 10, base_family = plot_family) +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 55, hjust = 1),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(face = "bold"))

save_plot(p_rho, "Figure_1_Spearman_heatmap", 13, 11)

cramers_v <- function(x, y) {
  tab <- table(x, y)
  if (min(dim(tab)) < 2) return(NA_real_)
  chi <- suppressWarnings(chisq.test(tab, correct = FALSE)$statistic)
  sqrt(as.numeric(chi) / (sum(tab) * (min(dim(tab)) - 1)))
}

vmat <- outer(factor_vars, factor_vars, Vectorize(\(a, b) {
  if (a == b) 1 else cramers_v(dat[[a]], dat[[b]])
}))
dimnames(vmat) <- list(factor_vars, factor_vars)

write_excel_csv(
  data.frame(variable = rownames(vmat), vmat, check.names = FALSE),
  file.path(out_dir, "04_cramers_v_matrix.csv")
)

v_long <- as.data.frame(as.table(vmat)) |>
  transmute(
    x = factor(Var1, levels = factor_vars),
    y = factor(Var2, levels = rev(factor_vars)),
    value = Freq,
    label = sprintf("%.2f", value)
  )

p_v <- ggplot(v_long, aes(x, y, fill = value)) +
  geom_tile(color = "white", linewidth = 0.2) +
  geom_text(aes(label = label), size = 3) +
  scale_fill_gradient(low = "white", high = "#2166AC",
                      limits = c(0, 1), name = "Cramer's V") +
  coord_fixed() +
  labs(title = "Association among categorical predictors", x = NULL, y = NULL) +
  theme_minimal(base_size = 10, base_family = plot_family) +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 55, hjust = 1),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(face = "bold"))

save_plot(p_v, "Figure_2_Cramers_V_heatmap", 10, 9)

# 4. LASSO design matrix --------------------------------------
fml <- reformulate(predictors, response = outcome)
term_obj <- terms(fml, data = dat)
x0 <- model.matrix(term_obj, dat)
assign0 <- attr(x0, "assign")
term_labels <- attr(term_obj, "term.labels")

keep <- colnames(x0) != "(Intercept)"
x <- x0[, keep, drop = FALSE]
term_index <- assign0[keep]

dummy_map <- tibble(
  dummy = colnames(x),
  predictor = term_labels[term_index]
)

# Remove accidental zero-variance dummy columns.
nonzero_var <- apply(x, 2, sd) > 0
x <- x[, nonzero_var, drop = FALSE]
dummy_map <- dummy_map[nonzero_var, ]
write_excel_csv(dummy_map, file.path(out_dir, "05_dummy_predictor_map.csv"))

make_foldid <- function(y, k, seed) {
  set.seed(seed)
  id <- integer(length(y))
  for (z in sort(unique(y))) {
    ii <- which(y == z)
    id[ii] <- sample(rep(seq_len(k), length.out = length(ii)))
  }
  id
}

extract_coef <- function(cvfit, s) {
  b <- as.matrix(coef(cvfit, s = s))
  tibble(dummy = rownames(b), coefficient = b[, 1]) |>
    filter(dummy != "(Intercept)", coefficient != 0) |>
    left_join(dummy_map, by = "dummy") |>
    arrange(desc(abs(coefficient)))
}

# A common lambda grid makes the repeated-CV results directly comparable.
lambda_grid <- glmnet(x, y, family = "binomial", alpha = 1,
                      standardize = TRUE, nlambda = 100)$lambda

# 5. Repeated stratified LASSO-CV ------------------------------
cvfits <- map(seq_len(n_repeats), \(r) {
  cv.glmnet(
    x, y,
    family = "binomial",
    alpha = 1,
    foldid = make_foldid(y, nfolds, 2025 + r),
    lambda = lambda_grid,
    type.measure = "deviance",
    standardize = TRUE,
    grouped = TRUE
  )
})

# The first fixed-seed run is the primary analysis.
# The remaining runs assess sensitivity to fold allocation.
primary <- cvfits[[1]]
selected_min <- extract_coef(primary, "lambda.min")
selected_1se <- extract_coef(primary, "lambda.1se")

cv_summary <- map_dfr(seq_along(cvfits), \(r) {
  a <- extract_coef(cvfits[[r]], "lambda.min")
  b <- extract_coef(cvfits[[r]], "lambda.1se")
  tibble(
    repeat_id = r,
    lambda_min = cvfits[[r]]$lambda.min,
    lambda_1se = cvfits[[r]]$lambda.1se,
    n_predictors_min = n_distinct(a$predictor),
    n_predictors_1se = n_distinct(b$predictor)
  )
})

selected_repeats <- map_dfr(seq_along(cvfits), \(r) {
  extract_coef(cvfits[[r]], "lambda.1se") |> mutate(repeat_id = r)
})

selection_frequency <- selected_repeats |>
  distinct(repeat_id, predictor) |>
  count(predictor, name = "selected_repeats") |>
  right_join(tibble(predictor = predictors), by = "predictor") |>
  mutate(
    selected_repeats = replace_na(selected_repeats, 0L),
    selection_frequency = selected_repeats / n_repeats
  ) |>
  arrange(desc(selection_frequency), predictor)

lambda_summary <- tibble(
  rule = c("lambda.min", "lambda.1se"),
  lambda = c(primary$lambda.min, primary$lambda.1se),
  n_dummy_terms = c(nrow(selected_min), nrow(selected_1se)),
  n_original_predictors = c(n_distinct(selected_min$predictor),
                            n_distinct(selected_1se$predictor))
)

write_excel_csv(lambda_summary, file.path(out_dir, "06_lambda_summary.csv"))
write_excel_csv(selected_min, file.path(out_dir, "07_selected_lambda_min.csv"))
write_excel_csv(selected_1se, file.path(out_dir, "08_selected_lambda_1se.csv"))
write_excel_csv(cv_summary, file.path(out_dir, "09_repeated_cv_summary.csv"))
write_excel_csv(selection_frequency,
                file.path(out_dir, "10_selection_frequency_lambda_1se.csv"))

# 6. LASSO figures --------------------------------------------
cv_curve <- tibble(
  lambda = primary$lambda,
  log_lambda = log(primary$lambda),
  deviance = primary$cvm,
  se = primary$cvsd,
  lower = deviance - se,
  upper = deviance + se,
  n_nonzero = primary$nzero
)
write_excel_csv(cv_curve, file.path(out_dir, "11_lasso_cv_curve.csv"))

p_cv <- ggplot(cv_curve, aes(log_lambda, deviance)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "#D9EAF4") +
  geom_line(color = "#145A86", linewidth = 0.9) +
  geom_point(color = "#145A86", size = 1.1) +
  geom_vline(xintercept = log(primary$lambda.min),
             linetype = 2, color = "#D55E00") +
  geom_vline(xintercept = log(primary$lambda.1se),
             linetype = 2, color = "#009E73") +
  labs(title = "LASSO cross-validation",
       subtitle = "Stratified 5-fold CV; binomial deviance; alpha = 1",
       x = "log(lambda)", y = "Cross-validated binomial deviance") +
  theme_classic(base_size = 11, base_family = plot_family) +
  theme(plot.title = element_text(face = "bold"))

save_plot(p_cv, "Figure_3_LASSO_CV", 7.5, 5.8)

pdf(file.path(out_dir, "Figure_4_LASSO_paths.pdf"), 8, 6,
    family = plot_family)
plot(primary$glmnet.fit, xvar = "lambda", label = FALSE,
     xlab = "log(lambda)", ylab = "Coefficient",
     main = "LASSO coefficient paths")
abline(v = log(primary$lambda.min), lty = 2, col = "#D55E00")
abline(v = log(primary$lambda.1se), lty = 2, col = "#009E73")
legend("topright", c("lambda.min", "lambda.1se"),
       lty = 2, col = c("#D55E00", "#009E73"), bty = "n")
dev.off()

freq_plot_data <- selection_frequency |>
  mutate(predictor = factor(predictor,
                            levels = predictor[order(selection_frequency)]))

p_freq <- ggplot(freq_plot_data,
                 aes(selection_frequency * 100, predictor)) +
  geom_col(fill = "#3B7EA1", width = 0.72) +
  geom_text(aes(label = sprintf("%.0f%%", selection_frequency * 100)),
            hjust = -0.1, size = 3) +
  scale_x_continuous(limits = c(0, 108), breaks = seq(0, 100, 20),
                     labels = \(x) paste0(x, "%"), expand = c(0, 0)) +
  labs(title = "Predictor stability at lambda.1se",
       subtitle = paste(n_repeats, "repeated stratified 5-fold CV runs"),
       x = "Selection frequency", y = NULL) +
  theme_classic(base_size = 10, base_family = plot_family) +
  theme(plot.title = element_text(face = "bold"),
        axis.text.y = element_text(size = 8))

save_plot(p_freq, "Figure_5_selection_frequency", 8.5, 8)

# 7. Save complete result object -------------------------------
results <- list(
  settings = list(outcome = outcome, positive = positive,
                  rho_cutoff = rho_cutoff, nfolds = nfolds,
                  n_repeats = n_repeats, predictors = predictors),
  audit = audit,
  spearman = rho,
  spearman_pairs = rho_pairs,
  cramers_v = vmat,
  primary_cv = primary,
  repeated_cv_summary = cv_summary,
  lambda_summary = lambda_summary,
  selected_lambda_min = selected_min,
  selected_lambda_1se = selected_1se,
  selection_frequency = selection_frequency
)
saveRDS(results, file.path(out_dir, "stage1_results.rds"))

summary_text <- c(
  paste0("Development n = ", nrow(dat)),
  paste0("Impacted = ", sum(y == 1), "; Non-impacted = ", sum(y == 0)),
  paste0("Candidate predictors = ", length(predictors)),
  paste0("High-correlation pairs |rho| >= ", rho_cutoff, ": ",
         sum(rho_pairs$abs_rho >= rho_cutoff)),
  paste0("lambda.min = ", signif(primary$lambda.min, 6)),
  paste0("lambda.1se = ", signif(primary$lambda.1se, 6)),
  paste0("Predictors at lambda.min = ", n_distinct(selected_min$predictor)),
  paste0("Predictors at lambda.1se = ", n_distinct(selected_1se$predictor)),
  paste0("lambda.1se predictors: ",
         paste(sort(unique(selected_1se$predictor)), collapse = ", "))
)
writeLines(summary_text, file.path(out_dir, "12_summary.txt"))
capture.output(sessionInfo(), file = file.path(out_dir, "13_sessionInfo.txt"))
cat(paste(summary_text, collapse = "\n"), "\n")
