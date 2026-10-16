# ============================================================
# Supplementary Table S2
# Nine algorithms: confusion matrices and classification metrics
# Based on patient-level nested out-of-fold predictions
# ============================================================

# install.packages(c("readr", "dplyr", "purrr"))

library(readr)
library(dplyr)
library(purrr)

# -----------------------------
# 1. Basic settings
# -----------------------------

pred_file <- "C:\\Users\\Lenovo\\Desktop\\impacted_pr\\output\\fully_nested_lasso_9models\\15_outer_predictions_all_models.csv"

positive <- "Impacted"
negative <- "Non-impacted"

# Common fixed classification threshold
threshold <- 0.50

# Bootstrap settings for uncertainty intervals of derived metrics
bootstrap_reps <- 2000L
bootstrap_seed <- 20261006L
bootstrap_conf_level <- 0.95

# The current file contains five outer predictions per patient and model
expected_outer_predictions <- 5

model_labels <- c(
  logistic      = "Logistic regression",
  elastic_net   = "Elastic net",
  random_forest = "Random forest",
  xgboost       = "XGBoost",
  lightgbm      = "LightGBM",
  svm_rbf       = "RBF-SVM",
  knn           = "KNN",
  mlp           = "MLP",
  decision_tree = "Decision tree"
)

model_order <- names(model_labels)

# -----------------------------
# 2. Read and validate raw data
# -----------------------------

raw <- read_csv(pred_file, show_col_types = FALSE)

required_columns <- c(
  "patient_id",
  "truth",
  "probability",
  "outer_id",
  "model"
)

missing_columns <- setdiff(required_columns, names(raw))

if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

raw <- raw %>%
  mutate(
    patient_id = as.character(patient_id),
    truth = trimws(as.character(truth)),
    model = trimws(as.character(model)),
    probability = as.numeric(probability)
  )

# Check outcome labels
observed_truth <- sort(unique(raw$truth))

if (!setequal(observed_truth, sort(c(positive, negative)))) {
  stop(
    "Unexpected truth labels: ",
    paste(observed_truth, collapse = ", ")
  )
}

# Check probabilities
if (any(is.na(raw$probability))) {
  stop("Missing prediction probabilities detected.")
}

if (any(raw$probability < 0 | raw$probability > 1)) {
  stop("Some probabilities are outside the range 0 to 1.")
}

# Check algorithms
observed_models <- sort(unique(raw$model))

if (!setequal(observed_models, sort(model_order))) {
  stop(
    "Observed models do not match the expected nine algorithms.\n",
    "Observed: ",
    paste(observed_models, collapse = ", ")
  )
}

# Check that each patient has only one observed outcome
truth_check <- raw %>%
  distinct(patient_id, truth) %>%
  count(patient_id, name = "n_truth_labels")

if (any(truth_check$n_truth_labels > 1)) {
  stop("At least one patient has inconsistent truth labels.")
}

# -----------------------------
# 3. Aggregate to patient level
# -----------------------------

patient_oof <- raw %>%
  group_by(patient_id, model) %>%
  summarise(
    truth = first(truth),
    probability = mean(probability, na.rm = TRUE),
    n_outer_predictions = n(),
    .groups = "drop"
  )

# Verify that every patient-model combination has five predictions
bad_repeats <- patient_oof %>%
  filter(n_outer_predictions != expected_outer_predictions)

if (nrow(bad_repeats) > 0) {
  stop(
    "Some patient-model combinations do not have ",
    expected_outer_predictions,
    " outer predictions."
  )
}

# Apply the common threshold after averaging probabilities
patient_oof <- patient_oof %>%
  mutate(
    predicted = if_else(
      probability >= threshold,
      positive,
      negative
    )
  )

# Save patient-level predictions for audit purposes
write_csv(
  patient_oof,
  "Table_S2_patient_level_nested_OOF_predictions.csv"
)

# -----------------------------
# 4. Metric functions
# -----------------------------

safe_divide <- function(numerator, denominator) {
  if (denominator == 0) {
    return(NA_real_)
  }
  numerator / denominator
}

exact_binomial_ci <- function(x, n, conf.level = 0.95) {
  if (n == 0) {
    return(c(NA_real_, NA_real_))
  }

  binom.test(
    x = x,
    n = n,
    conf.level = conf.level
  )$conf.int
}

safe_logit <- function(p, eps = 1e-6) {
  p <- pmin(pmax(as.numeric(p), eps), 1 - eps)
  qlogis(p)
}

# Rank-based AUC; avoids requiring pROC for this supplementary table.
auc_rank <- function(y, p) {
  y <- as.integer(y)
  p <- as.numeric(p)
  n_pos <- sum(y == 1L)
  n_neg <- sum(y == 0L)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  ranks <- rank(p, ties.method = "average")
  (sum(ranks[y == 1L]) - n_pos * (n_pos + 1) / 2) /
    (n_pos * n_neg)
}

calibration_intercept <- function(y, p) {
  fit <- tryCatch(
    suppressWarnings(
      glm(y ~ 1, offset = safe_logit(p), family = binomial())
    ),
    error = function(e) NULL
  )
  if (is.null(fit) || !isTRUE(fit$converged)) return(NA_real_)
  unname(coef(fit)[1])
}

calibration_slope <- function(y, p) {
  lp <- safe_logit(p)
  fit <- tryCatch(
    suppressWarnings(
      glm(y ~ lp, family = binomial())
    ),
    error = function(e) NULL
  )
  if (is.null(fit) || !isTRUE(fit$converged)) return(NA_real_)
  unname(coef(fit)[2])
}

derived_metrics <- function(y, p, threshold = 0.50) {
  y <- as.integer(y)
  p <- as.numeric(p)
  pred <- p >= threshold

  # na.rm = TRUE prevents a single malformed row from propagating NA
  # into the bootstrap calculation.
  tp <- sum(y == 1L & pred, na.rm = TRUE)
  fn <- sum(y == 1L & !pred, na.rm = TRUE)
  tn <- sum(y == 0L & !pred, na.rm = TRUE)
  fp <- sum(y == 0L & pred, na.rm = TRUE)

  sensitivity <- safe_divide(tp, tp + fn)
  specificity <- safe_divide(tn, tn + fp)
  ppv <- safe_divide(tp, tp + fp)
  npv <- safe_divide(tn, tn + fn)
  f1 <- safe_divide(2 * ppv * sensitivity, ppv + sensitivity)
  accuracy <- safe_divide(tp + tn, length(y))
  youden <- sensitivity + specificity - 1
  mcc_den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- if (!length(mcc_den) || !is.finite(mcc_den) || mcc_den == 0) {
    NA_real_
  } else {
    (tp * tn - fp * fn) / mcc_den
  }

  c(
    ROC_AUC = auc_rank(y, p),
    F1_score = f1,
    MCC = mcc,
    Youden_index = youden,
    Brier_score = mean((p - y)^2),
    Calibration_intercept = calibration_intercept(y, p),
    Calibration_slope = calibration_slope(y, p)
  )
}

bootstrap_derived_ci <- function(y, p, threshold = 0.50,
                                 reps = 2000L, seed = 1L,
                                 conf.level = 0.95) {
  y <- as.integer(y)
  p <- as.numeric(p)
  pos <- which(y == 1L)
  neg <- which(y == 0L)
  if (!length(pos) || !length(neg)) {
    return(matrix(NA_real_, nrow = 7L, ncol = 2L,
                  dimnames = list(c(
                    "ROC_AUC", "F1_score", "MCC", "Youden_index",
                    "Brier_score", "Calibration_intercept",
                    "Calibration_slope"
                  ), c("LCL95", "UCL95"))))
  }

  set.seed(seed)
  metric_names <- c(
    "ROC_AUC", "F1_score", "MCC", "Youden_index", "Brier_score",
    "Calibration_intercept", "Calibration_slope"
  )
  boot_values <- matrix(NA_real_, nrow = reps, ncol = length(metric_names),
                        dimnames = list(NULL, metric_names))

  for (b in seq_len(reps)) {
    idx <- c(
      sample(pos, length(pos), replace = TRUE),
      sample(neg, length(neg), replace = TRUE)
    )
    boot_values[b, ] <- derived_metrics(y[idx], p[idx], threshold)
  }

  alpha <- 1 - conf.level
  ci <- t(apply(
    boot_values,
    2,
    function(x) {
      x <- x[is.finite(x)]
      if (!length(x)) return(c(NA_real_, NA_real_))
      as.numeric(quantile(x, probs = c(alpha / 2, 1 - alpha / 2),
                          names = FALSE, type = 7, na.rm = TRUE))
    }
  ))
  colnames(ci) <- c("LCL95", "UCL95")
  ci
}

calculate_metrics <- function(data, model_id) {

  observed <- data$truth == positive
  predicted <- data$predicted == positive

  TP <- sum(observed & predicted)
  FN <- sum(observed & !predicted)
  TN <- sum(!observed & !predicted)
  FP <- sum(!observed & predicted)

  N <- TP + FN + TN + FP

  sensitivity <- safe_divide(TP, TP + FN)
  specificity <- safe_divide(TN, TN + FP)
  ppv <- safe_divide(TP, TP + FP)
  npv <- safe_divide(TN, TN + FN)

  f1_score <- safe_divide(
    2 * ppv * sensitivity,
    ppv + sensitivity
  )

  accuracy <- safe_divide(TP + TN, N)
  youden_index <- sensitivity + specificity - 1

  sensitivity_ci <- exact_binomial_ci(TP, TP + FN)
  specificity_ci <- exact_binomial_ci(TN, TN + FP)
  ppv_ci <- exact_binomial_ci(TP, TP + FP)
  npv_ci <- exact_binomial_ci(TN, TN + FN)
  accuracy_ci <- exact_binomial_ci(TP + TN, N)

  y <- as.integer(observed)
  p <- data$probability
  point_derived <- derived_metrics(y, p, threshold)
  derived_ci <- bootstrap_derived_ci(
    y = y,
    p = p,
    threshold = threshold,
    reps = bootstrap_reps,
    seed = bootstrap_seed + match(model_id, model_order),
    conf.level = bootstrap_conf_level
  )

  tibble(
    Model_ID = model_id,
    Algorithm = unname(model_labels[model_id]),
    Threshold = threshold,

    N = N,
    TP = TP,
    FN = FN,
    TN = TN,
    FP = FP,

    Sensitivity = sensitivity,
    Sensitivity_LCL95 = sensitivity_ci[1],
    Sensitivity_UCL95 = sensitivity_ci[2],

    Specificity = specificity,
    Specificity_LCL95 = specificity_ci[1],
    Specificity_UCL95 = specificity_ci[2],

    PPV = ppv,
    PPV_LCL95 = ppv_ci[1],
    PPV_UCL95 = ppv_ci[2],

    NPV = npv,
    NPV_LCL95 = npv_ci[1],
    NPV_UCL95 = npv_ci[2],

    F1_score = f1_score,
    F1_score_LCL95 = derived_ci["F1_score", "LCL95"],
    F1_score_UCL95 = derived_ci["F1_score", "UCL95"],

    Accuracy = accuracy,
    Accuracy_LCL95 = accuracy_ci[1],
    Accuracy_UCL95 = accuracy_ci[2],

    Youden_index = youden_index,
    Youden_index_LCL95 = derived_ci["Youden_index", "LCL95"],
    Youden_index_UCL95 = derived_ci["Youden_index", "UCL95"],

    MCC = point_derived["MCC"],
    MCC_LCL95 = derived_ci["MCC", "LCL95"],
    MCC_UCL95 = derived_ci["MCC", "UCL95"],

    ROC_AUC = point_derived["ROC_AUC"],
    ROC_AUC_LCL95 = derived_ci["ROC_AUC", "LCL95"],
    ROC_AUC_UCL95 = derived_ci["ROC_AUC", "UCL95"],

    Brier_score = point_derived["Brier_score"],
    Brier_score_LCL95 = derived_ci["Brier_score", "LCL95"],
    Brier_score_UCL95 = derived_ci["Brier_score", "UCL95"],

    Calibration_intercept = point_derived["Calibration_intercept"],
    Calibration_intercept_LCL95 = derived_ci["Calibration_intercept", "LCL95"],
    Calibration_intercept_UCL95 = derived_ci["Calibration_intercept", "UCL95"],

    Calibration_slope = point_derived["Calibration_slope"],
    Calibration_slope_LCL95 = derived_ci["Calibration_slope", "LCL95"],
    Calibration_slope_UCL95 = derived_ci["Calibration_slope", "UCL95"]
  )
}

# -----------------------------
# 5. Generate Table S2
# -----------------------------

table_s2 <- map_dfr(
  model_order,
  ~ calculate_metrics(
    data = patient_oof %>% filter(model == .x),
    model_id = .x
  )
)

# Round continuous metrics for presentation
table_s2_display <- table_s2 %>%
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 4)
    )
  )

# Full table with 95% confidence intervals
write_csv(
  table_s2_display,
  "Table_S2_nine_algorithms_confusion_matrix_and_classification_metrics.csv"
)

# Main table for manuscript display
table_s2_main <- table_s2_display %>%
  select(
    Algorithm,
    Model_ID,
    Threshold,
    N,
    TP,
    FN,
    TN,
    FP,
    Sensitivity,
    Sensitivity_LCL95,
    Sensitivity_UCL95,
    Specificity,
    Specificity_LCL95,
    Specificity_UCL95,
    PPV,
    PPV_LCL95,
    PPV_UCL95,
    NPV,
    NPV_LCL95,
    NPV_UCL95,
    F1_score,
    F1_score_LCL95,
    F1_score_UCL95,
    Accuracy,
    Accuracy_LCL95,
    Accuracy_UCL95,
    MCC,
    MCC_LCL95,
    MCC_UCL95,
    ROC_AUC,
    ROC_AUC_LCL95,
    ROC_AUC_UCL95,
    Brier_score,
    Brier_score_LCL95,
    Brier_score_UCL95,
    Calibration_intercept,
    Calibration_intercept_LCL95,
    Calibration_intercept_UCL95,
    Calibration_slope,
    Calibration_slope_LCL95,
    Calibration_slope_UCL95,
    Youden_index,
    Youden_index_LCL95,
    Youden_index_UCL95
  )

write_csv(
  table_s2_main,
  "Table_S2_nine_algorithms_main_results.csv"
)

print(table_s2_main)

getwd()

# Suggested table footnote:
# The positive event was uniformly defined as impacted stone for all nine
# algorithms. Classification metrics were calculated from patient-level nested
# out-of-fold predictions. For repeated outer resampling, predicted
# probabilities were averaged within patient before applying a common
# classification threshold of 0.50. Confidence intervals for sensitivity,
# specificity, PPV, NPV, and accuracy were calculated using the exact binomial
# method. Confidence intervals for ROC AUC, F1 score, MCC, Youden index, Brier
# score, calibration intercept, and calibration slope were calculated from
# 2,000 patient-level stratified bootstrap resamples. The calibration
# intercept was estimated from a logistic model with the logit predicted
# probability as an offset; the calibration slope was estimated from a
# logistic model with the logit predicted probability as the only covariate.
