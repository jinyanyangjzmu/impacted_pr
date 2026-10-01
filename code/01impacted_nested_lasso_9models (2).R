# ============================================================
# Impacted stone study
# Nested LASSO feature selection + nine-model comparison
#
# Development cohort only. The external cohort is not read here.
# Outcome event: Impacted
# Outer validation: stratified 5-fold CV repeated 5 times
# Inner tuning: stratified 5-fold CV
# Model selection criterion: mean inner-CV ROC AUC
# Usage: Rscript impacted_nested_lasso_9models.R [data.rds] [output_dir]
# ============================================================

# 1. Packages -------------------------------------------------
required_pkgs <- c(
  "tidymodels", "glmnet", "ranger", "kernlab", "kknn",
  "bonsai", "lightgbm", "nnet", "xgboost", "doParallel", "readr"
)

missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_pkgs)) {
  stop("Install required packages first: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(tidymodels)
  library(glmnet)
  library(bonsai)
  library(doParallel)
  library(readr)
})

tidymodels_prefer()

# 2. Settings -------------------------------------------------
script_version <- "1.1.0"
seed <- 2026
RNGkind("L'Ecuyer-CMRG")
outer_v <- 5
outer_repeats <- 5
inner_v <- 5
auc_closeness <- 0.01
grid_design <- "max_min_l2"
boundary_alert_threshold <- 0.20

# More complex search spaces receive larger, prespecified tuning budgets.
# Logistic regression has no hyperparameters.
grid_sizes <- c(
  logistic = 0L,
  elastic_net = 25L,
  decision_tree = 30L,
  random_forest = 30L,
  svm_rbf = 30L,
  knn = 30L,
  lightgbm = 50L,
  mlp = 35L,
  xgboost = 50L
)

scale_sensitive_models <- c("elastic_net", "svm_rbf", "knn", "mlp")

outcome <- "impacted_stone"
id_var <- "patient_id"
positive <- "Impacted"
negative <- "Non-impacted"

# Prespecified redundancy exclusions. These variables are always removed
# before LASSO; this is not an outcome-driven step.
correlation_drop <- c(
  "stone_width",
  "hu_above_below_ratio",
  "alcohol_use",
  "hydronephrosis_grade",
  "ureteral_wall_area"
)

args <- commandArgs(trailingOnly = TRUE)
default_data_path <- "C:/Users/Lenovo/Desktop/impacted_pr/data/impacted_clean.rds"
data_path <- if (length(args) >= 1L) {
  args[[1]]
} else {
  Sys.getenv("IMPACTED_DATA_PATH", unset = default_data_path)
}
if (!file.exists(data_path)) {
  stop(
    "Data file not found: ", data_path,
    "\nPass the RDS path as the first command-line argument or set ",
    "IMPACTED_DATA_PATH."
  )
}
data_path <- normalizePath(data_path, winslash = "/", mustWork = TRUE)
data_md5 <- unname(tools::md5sum(data_path))

project_dir <- if (basename(dirname(data_path)) == "data") {
  dirname(dirname(data_path))
} else {
  dirname(data_path)
}
out_dir <- if (length(args) >= 2L) {
  args[[2]]
} else {
  file.path(project_dir, "output", "nested_lasso_9models_publication")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, winslash = "/", mustWork = TRUE)

model_labels <- c(
  logistic = "Logistic regression",
  elastic_net = "Elastic net",
  decision_tree = "Decision tree",
  random_forest = "Random forest",
  svm_rbf = "RBF-SVM",
  knn = "k-nearest neighbors",
  lightgbm = "LightGBM",
  mlp = "Multilayer perceptron",
  xgboost = "XGBoost"
)

model_tuning_parameters <- c(
  logistic = "None",
  elastic_net = "penalty; mixture",
  decision_tree = "cost_complexity; tree_depth; min_n",
  random_forest = "mtry; min_n (trees fixed at 1000)",
  svm_rbf = "cost; rbf_sigma",
  knn = "neighbors; weight_func; dist_power",
  lightgbm = paste(
    "mtry; trees; min_n; tree_depth; learn_rate;",
    "loss_reduction; sample_size"
  ),
  mlp = "hidden_units; penalty; epochs",
  xgboost = paste(
    "mtry; trees; min_n; tree_depth; learn_rate;",
    "loss_reduction; sample_size"
  )
)

if (!identical(names(grid_sizes), names(model_labels))) {
  stop("grid_sizes and model_labels must contain the same models in the same order.")
}
if (!identical(names(model_tuning_parameters), names(model_labels))) {
  stop("model_tuning_parameters and model_labels must have matching names.")
}

# 3. Data -----------------------------------------------------
dat <- readRDS(data_path) |> as.data.frame(check.names = FALSE)

required <- c(id_var, outcome, correlation_drop)
if (!all(required %in% names(dat))) {
  stop("Missing required columns: ",
       paste(setdiff(required, names(dat)), collapse = ", "))
}
if (anyNA(dat)) stop("The dataset contains missing values.")
if (anyDuplicated(dat[[id_var]])) stop("patient_id must be unique.")

dat[[outcome]] <- factor(
  trimws(as.character(dat[[outcome]])),
  levels = c(positive, negative)
)
if (anyNA(dat[[outcome]])) stop("Unexpected outcome labels.")
if (any(table(dat[[outcome]]) == 0L)) stop("Both outcome classes are required.")

to_factor <- names(dat)[
  vapply(dat, function(x) is.character(x) || is.logical(x), logical(1))
]
dat[to_factor] <- lapply(dat[to_factor], factor)

candidate_predictors <- setdiff(
  names(dat), c(id_var, outcome, correlation_drop)
)
if (!length(candidate_predictors)) stop("No candidate predictors were found.")

nonfinite_numeric <- names(dat)[
  vapply(
    dat,
    function(x) is.numeric(x) && any(!is.finite(x)),
    logical(1)
  )
]
if (length(nonfinite_numeric)) {
  stop(
    "Non-finite numeric values found in: ",
    paste(nonfinite_numeric, collapse = ", ")
  )
}

audit <- tibble(
  item = c(
    "patients", "impacted", "non_impacted", "original_predictors",
    "prespecified_drops", "lasso_candidates", "outer_resamples"
  ),
  value = c(
    nrow(dat), sum(dat[[outcome]] == positive),
    sum(dat[[outcome]] == negative), ncol(dat) - 2,
    length(correlation_drop), length(candidate_predictors),
    outer_v * outer_repeats
  )
)

# 4. Helper functions ----------------------------------------
make_foldid <- function(y, v, seed_value) {
  set.seed(seed_value)
  fold_id <- integer(length(y))
  for (level in sort(unique(y))) {
    index <- which(y == level)
    fold_id[index] <- sample(rep(seq_len(v), length.out = length(index)))
  }
  fold_id
}

select_lasso_variables <- function(train_data, predictors, seed_value) {
  formula <- reformulate(predictors, response = outcome)
  terms_obj <- terms(formula, data = train_data)
  matrix_all <- model.matrix(terms_obj, train_data)
  assignment <- attr(matrix_all, "assign")
  term_labels <- attr(terms_obj, "term.labels")

  keep <- colnames(matrix_all) != "(Intercept)"
  x <- matrix_all[, keep, drop = FALSE]
  dummy_map <- tibble(
    dummy = colnames(x),
    predictor = term_labels[assignment[keep]]
  )

  nonzero_variance <- apply(x, 2, sd) > 0
  x <- x[, nonzero_variance, drop = FALSE]
  dummy_map <- dummy_map[nonzero_variance, , drop = FALSE]

  if (!ncol(x)) stop("No non-zero-variance predictors remained for LASSO.")

  y <- as.integer(train_data[[outcome]] == positive)
  fold_id <- make_foldid(y, inner_v, seed_value)

  set.seed(seed_value)
  cv_fit <- cv.glmnet(
    x = x,
    y = y,
    family = "binomial",
    alpha = 1,
    foldid = fold_id,
    type.measure = "deviance",
    standardize = TRUE,
    grouped = TRUE
  )

  nonzero_dummies <- function(lambda_value) {
    coefficients <- as.matrix(coef(cv_fit, s = lambda_value))
    rownames(coefficients)[
      rownames(coefficients) != "(Intercept)" & coefficients[, 1] != 0
    ]
  }

  lambda_used <- cv_fit$lambda.1se
  lambda_rule <- "lambda.1se"
  selected_dummies <- nonzero_dummies(lambda_used)

  if (!length(selected_dummies)) {
    lambda_used <- cv_fit$lambda.min
    lambda_rule <- "lambda.min_fallback"
    selected_dummies <- nonzero_dummies(lambda_used)
  }
  if (!length(selected_dummies)) {
    stop("LASSO selected no variables at lambda.min.")
  }

  selected <- dummy_map |>
    filter(dummy %in% selected_dummies) |>
    pull(predictor) |>
    unique()

  # Restore the original predictor order and retain a complete factor when
  # any of its dummy variables was selected.
  selected <- intersect(predictors, selected)

  list(
    selected = selected,
    lambda_rule = lambda_rule,
    lambda_used = lambda_used,
    lambda_1se = cv_fit$lambda.1se,
    lambda_min = cv_fit$lambda.min,
    n_selected = length(selected),
    n_selected_dummies = length(selected_dummies)
  )
}

count_design_columns <- function(data, predictors) {
  x <- model.matrix(reformulate(predictors), data)
  x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  if (ncol(x)) x <- x[, apply(x, 2, sd) > 0, drop = FALSE]
  max(1L, ncol(x))
}

make_recipe <- function(train_data, predictors, normalize = FALSE) {
  rec <- recipe(
    reformulate(predictors, response = outcome), data = train_data
  ) |>
    step_unknown(all_nominal_predictors(), new_level = "unknown") |>
    step_novel(all_nominal_predictors(), new_level = "novel") |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors())

  if (normalize) rec <- rec |> step_normalize(all_numeric_predictors())
  rec
}

make_models <- function(p) {
  mtry_value <- if (p > 1) tune() else 1L
  mtry_parameter <- if (p > 1) list(dials::mtry(c(1L, p))) else list()
  parameter_set <- function(...) do.call(dials::parameters, list(...))

  boost_definition <- function(engine, min_n_range) {
    spec <- boost_tree(
      mtry = !!mtry_value, trees = tune(), min_n = tune(),
      tree_depth = tune(), learn_rate = tune(),
      loss_reduction = tune(), sample_size = tune()
    )
    spec <- if (engine == "lightgbm") {
      set_engine(spec, "lightgbm", num_threads = 1, verbose = -1)
    } else {
      set_engine(spec, "xgboost", nthread = 1, verbose = 0)
    }

    list(
      spec = set_mode(spec, "classification"),
      params = do.call(
        dials::parameters,
        c(mtry_parameter, list(
          dials::trees(c(200L, 1200L)),
          dials::min_n(min_n_range),
          dials::tree_depth(c(2L, 8L)),
          dials::learn_rate(c(-3, -0.3)),
          dials::loss_reduction(c(-6, 1)),
          dials::sample_prop(c(0.60, 1.00))
        ))
      )
    )
  }

  list(
    logistic = list(
      spec = logistic_reg() |>
        set_engine("glm") |>
        set_mode("classification"),
      params = NULL
    ),
    elastic_net = list(
      spec = logistic_reg(penalty = tune(), mixture = tune()) |>
        set_engine("glmnet") |>
        set_mode("classification"),
      params = parameter_set(
        dials::penalty(c(-5, 0)),
        dials::mixture(c(0, 1))
      )
    ),
    decision_tree = list(
      spec = decision_tree(
        cost_complexity = tune(), tree_depth = tune(), min_n = tune()
      ) |>
        set_engine("rpart") |>
        set_mode("classification"),
      params = parameter_set(
        dials::cost_complexity(c(-5, -1)),
        dials::tree_depth(c(2L, 10L)),
        dials::min_n(c(5L, 40L))
      )
    ),
    random_forest = list(
      spec = rand_forest(
        mtry = !!mtry_value, min_n = tune(), trees = 1000
      ) |>
        set_engine("ranger", probability = TRUE, num.threads = 1) |>
        set_mode("classification"),
      params = do.call(
        dials::parameters,
        c(mtry_parameter, list(dials::min_n(c(2L, 30L))))
      )
    ),
    svm_rbf = list(
      spec = svm_rbf(cost = tune(), rbf_sigma = tune()) |>
        set_engine("kernlab") |>
        set_mode("classification"),
      params = parameter_set(
        dials::cost(c(-3, 3)),
        dials::rbf_sigma(c(-5, 0))
      )
    ),
    knn = list(
      spec = nearest_neighbor(
        neighbors = tune(), weight_func = tune(), dist_power = tune()
      ) |>
        set_engine("kknn") |>
        set_mode("classification"),
      params = parameter_set(
        dials::neighbors(c(3L, 35L)),
        dials::weight_func(
          values = c("rectangular", "triangular", "epanechnikov")
        ),
        dials::dist_power(c(1, 2))
      )
    ),
    lightgbm = boost_definition("lightgbm", c(5L, 40L)),
    mlp = list(
      spec = mlp(
        hidden_units = tune(), penalty = tune(), epochs = tune()
      ) |>
        set_engine("nnet", trace = FALSE, MaxNWts = 10000) |>
        set_mode("classification"),
      params = parameter_set(
        dials::hidden_units(c(2L, 20L)),
        dials::penalty(c(-6, -1)),
        dials::epochs(c(50L, 300L))
      )
    ),
    xgboost = boost_definition("xgboost", c(2L, 30L))
  )
}

fit_one_model <- function(
    model_name, definition, model_recipe, train_data, assessment_data,
    inner_folds, outer_id, seed_value) {

  workflow <- workflow() |>
    add_recipe(model_recipe) |>
    add_model(definition$spec)

  inner_auc <- NA_real_
  tuning_notes <- tibble()
  tuning_grid <- tibble()
  n_grid <- 0L

  if (is.null(definition$params)) {
    final_workflow <- workflow
    best_parameters <- tibble()
  } else {
    set.seed(seed_value)
    grid <- dials::grid_space_filling(
      definition$params,
      size = unname(grid_sizes[[model_name]]),
      type = grid_design
    )
    n_grid <- nrow(grid)
    tuning_grid <- grid |>
      mutate(
        outer_id = outer_id,
        model = model_name,
        candidate_id = row_number(),
        .before = 1
      )

    tuned <- tune_grid(
      workflow,
      resamples = inner_folds,
      grid = grid,
      metrics = metric_set(roc_auc, brier_class),
      control = control_grid(
        save_pred = FALSE,
        save_workflow = FALSE,
        verbose = FALSE,
        allow_par = TRUE,
        event_level = "first",
        parallel_over = "resamples",
        pkgs = required_pkgs
      )
    )

    metric_table <- collect_metrics(tuned)
    if (!any(metric_table$.metric == "roc_auc" & is.finite(metric_table$mean))) {
      stop("No valid inner-CV ROC AUC estimates.")
    }

    best_parameters <- select_best(tuned, metric = "roc_auc")
    inner_auc <- show_best(tuned, metric = "roc_auc", n = 1)$mean[[1]]
    tuning_notes <- collect_notes(tuned) |>
      mutate(outer_id = outer_id, model = model_name, .before = 1)
    final_workflow <- finalize_workflow(workflow, best_parameters)
  }

  set.seed(seed_value)
  fitted <- fit(final_workflow, data = train_data)
  probability_column <- paste0(".pred_", make.names(positive))
  probability <- predict(fitted, assessment_data, type = "prob")

  if (!probability_column %in% names(probability)) {
    stop("Positive-class probability column was not found.")
  }
  probability_value <- probability[[probability_column]]
  if (any(!is.finite(probability_value)) ||
      any(probability_value < 0 | probability_value > 1)) {
    stop("Model returned invalid probabilities.")
  }

  predictions <- tibble(
    patient_id = assessment_data[[id_var]],
    truth = assessment_data[[outcome]],
    probability = probability_value,
    outer_id = outer_id,
    model = model_name
  )

  parameter_row <- tibble(
    outer_id = outer_id,
    model = model_name,
    inner_best_auc = inner_auc,
    tuning_candidates = n_grid,
    normalized_predictors = model_name %in% scale_sensitive_models
  )
  if (nrow(best_parameters)) {
    parameter_row <- bind_cols(parameter_row, best_parameters)
  }

  list(
    predictions = predictions,
    parameters = parameter_row,
    tuning_grid = tuning_grid,
    notes = tuning_notes
  )
}

safe_auc <- function(truth, probability) {
  tryCatch(
    roc_auc_vec(truth, probability, event_level = "first"),
    error = function(e) NA_real_
  )
}

calibration_metrics <- function(truth, probability) {
  y <- as.integer(truth == positive)
  probability <- pmin(pmax(probability, 1e-6), 1 - 1e-6)
  lp <- qlogis(probability)

  intercept <- tryCatch(
    unname(coef(glm(y ~ offset(lp), family = binomial()))[1]),
    error = function(e) NA_real_
  )
  slope <- tryCatch(
    unname(coef(glm(y ~ lp, family = binomial()))[2]),
    error = function(e) NA_real_
  )

  tibble(calibration_intercept = intercept, calibration_slope = slope)
}

save_plot <- function(plot, filename, width, height) {
  pdf_device <- if (capabilities("cairo")) {
    grDevices::cairo_pdf
  } else {
    grDevices::pdf
  }
  ggsave(
    file.path(out_dir, paste0(filename, ".pdf")), plot,
    width = width, height = height, device = pdf_device
  )
  ggsave(
    file.path(out_dir, paste0(filename, ".png")), plot,
    width = width, height = height, dpi = 600, bg = "white"
  )
}

write_table <- function(data, filename) {
  if (!ncol(data)) data <- tibble(message = "No records")
  write_excel_csv(data, file.path(out_dir, filename))
}

# 5. Nested resampling ----------------------------------------
run_analysis <- function() {
  available_cores <- parallel::detectCores(logical = FALSE)
  if (is.na(available_cores)) available_cores <- 2L
  n_cores <- max(1L, min(8L, available_cores - 1L))

  cluster <- parallel::makePSOCKcluster(n_cores)
  doParallel::registerDoParallel(cluster)
  on.exit({
    parallel::stopCluster(cluster)
    foreach::registerDoSEQ()
  }, add = TRUE)

  set.seed(seed)
  outer_folds <- vfold_cv(
    dat, v = outer_v, repeats = outer_repeats, strata = impacted_stone
  )

  prediction_log <- list()
  parameter_log <- list()
  grid_log <- list()
  note_log <- list()
  error_log <- list()
  lasso_summary_log <- list()
  lasso_variable_log <- list()
  membership_log <- list()

  for (i in seq_len(nrow(outer_folds))) {
    outer_id <- if ("id2" %in% names(outer_folds)) {
      paste(outer_folds$id[[i]], outer_folds$id2[[i]], sep = "_")
    } else {
      outer_folds$id[[i]]
    }

    message("Outer resample ", i, "/", nrow(outer_folds), ": ", outer_id)
    split <- outer_folds$splits[[i]]
    train_outer <- analysis(split)
    assess_outer <- assessment(split)

    membership_log[[outer_id]] <- tibble(
      outer_id = outer_id,
      patient_id = assess_outer[[id_var]],
      role = "assessment"
    )

    lasso_result <- tryCatch(
      select_lasso_variables(
        train_outer, candidate_predictors, seed + 10000L + i
      ),
      error = function(e) e
    )

    if (inherits(lasso_result, "error")) {
      error_log[[paste0(outer_id, "_lasso")]] <- tibble(
        outer_id = outer_id,
        model = "LASSO",
        stage = "feature_selection",
        error = conditionMessage(lasso_result)
      )
      next
    }

    selected <- lasso_result$selected
    lasso_summary_log[[outer_id]] <- tibble(
      outer_id = outer_id,
      lambda_rule = lasso_result$lambda_rule,
      lambda_used = lasso_result$lambda_used,
      lambda_1se = lasso_result$lambda_1se,
      lambda_min = lasso_result$lambda_min,
      n_selected = lasso_result$n_selected,
      n_selected_dummies = lasso_result$n_selected_dummies
    )
    lasso_variable_log[[outer_id]] <- tibble(
      outer_id = outer_id,
      predictor = selected
    )

    train_model <- train_outer[, c(id_var, outcome, selected), drop = FALSE]
    assess_model <- assess_outer[, c(id_var, outcome, selected), drop = FALSE]
    p_design <- count_design_columns(train_model, selected)
    definitions <- make_models(p_design)
    if (!identical(names(definitions), names(model_labels))) {
      stop("Model definitions and model_labels are not aligned.")
    }
    if (any(table(train_model[[outcome]]) < inner_v)) {
      stop("An outcome class has fewer observations than inner_v.")
    }

    set.seed(seed + 20000L + i)
    inner_folds <- vfold_cv(
      train_model, v = inner_v, strata = impacted_stone
    )

    for (j in seq_along(definitions)) {
      model_name <- names(definitions)[[j]]
      message("  ", model_labels[[model_name]])
      model_recipe <- make_recipe(
        train_model,
        selected,
        normalize = model_name %in% scale_sensitive_models
      )

      result <- tryCatch(
        fit_one_model(
          model_name = model_name,
          definition = definitions[[model_name]],
          model_recipe = model_recipe,
          train_data = train_model,
          assessment_data = assess_model,
          inner_folds = inner_folds,
          outer_id = outer_id,
          seed_value = seed + 30000L + i * 100L + j
        ),
        error = function(e) e
      )

      key <- paste(outer_id, model_name, sep = "_")
      if (inherits(result, "error")) {
        message("    FAILED: ", conditionMessage(result))
        error_log[[key]] <- tibble(
          outer_id = outer_id,
          model = model_name,
          stage = "tuning_or_fit",
          error = conditionMessage(result)
        )
      } else {
        prediction_log[[key]] <- result$predictions
        parameter_log[[key]] <- result$parameters
        if (nrow(result$tuning_grid)) grid_log[[key]] <- result$tuning_grid
        if (nrow(result$notes)) note_log[[key]] <- result$notes
      }
    }

    saveRDS(
      list(
        predictions = bind_rows(prediction_log),
        lasso_summary = bind_rows(lasso_summary_log),
        lasso_variables = bind_rows(lasso_variable_log),
        best_parameters = bind_rows(parameter_log),
        tuning_grids = bind_rows(grid_log),
        errors = bind_rows(error_log)
      ),
      file.path(out_dir, "checkpoint_nested_cv.rds")
    )
  }

  list(
    outer_folds = outer_folds,
    predictions = bind_rows(prediction_log),
    parameters = bind_rows(parameter_log),
    tuning_grids = bind_rows(grid_log),
    notes = bind_rows(note_log),
    errors = bind_rows(error_log),
    lasso_summary = bind_rows(lasso_summary_log),
    lasso_variables = bind_rows(lasso_variable_log),
    membership = bind_rows(membership_log),
    n_cores = n_cores
  )
}

raw_results <- run_analysis()

if (!nrow(raw_results$predictions)) {
  stop("No outer-assessment predictions were produced. Check the error log.")
}

# 6. Outer-CV performance ------------------------------------
fold_metrics <- raw_results$predictions |>
  group_by(outer_id, model) |>
  summarise(
    n = n(),
    roc_auc = safe_auc(truth, probability),
    brier = mean((as.integer(truth == positive) - probability)^2),
    .groups = "drop"
  )

# Each patient has one assessment prediction per repeat. Average the five
# repeated-CV probabilities before calculating pooled OOF performance.
patient_oof <- raw_results$predictions |>
  group_by(patient_id, model, truth) |>
  summarise(
    probability = mean(probability),
    n_predictions = n(),
    .groups = "drop"
  )

prediction_coverage <- patient_oof |>
  group_by(model) |>
  summarise(
    covered_patients = n_distinct(patient_id),
    minimum_predictions_per_patient = min(n_predictions),
    maximum_predictions_per_patient = max(n_predictions),
    complete_oof_coverage = (
      covered_patients == nrow(dat) &
        minimum_predictions_per_patient == outer_repeats &
        maximum_predictions_per_patient == outer_repeats
    ),
    .groups = "drop"
  )

pooled_metrics <- patient_oof |>
  group_by(model) |>
  group_modify(~ {
    complete <- nrow(.x) == nrow(dat) &&
      all(.x$n_predictions == outer_repeats)
    if (!complete) {
      return(tibble(
        pooled_oof_auc = NA_real_,
        pooled_oof_brier = NA_real_,
        calibration_intercept = NA_real_,
        calibration_slope = NA_real_,
        n_patients = nrow(.x),
        complete_oof_coverage = FALSE
      ))
    }

    calibration <- calibration_metrics(.x$truth, .x$probability)
    tibble(
      pooled_oof_auc = safe_auc(.x$truth, .x$probability),
      pooled_oof_brier = mean(
        (as.integer(.x$truth == positive) - .x$probability)^2
      ),
      calibration_intercept = calibration$calibration_intercept,
      calibration_slope = calibration$calibration_slope,
      n_patients = nrow(.x),
      complete_oof_coverage = TRUE
    )
  }) |>
  ungroup()

model_summary <- fold_metrics |>
  group_by(model) |>
  summarise(
    completed_outer_folds = sum(is.finite(roc_auc)),
    outer_auc_mean = mean(roc_auc, na.rm = TRUE),
    outer_auc_sd = sd(roc_auc, na.rm = TRUE),
    outer_brier_mean = mean(brier, na.rm = TRUE),
    .groups = "drop"
  ) |>
  right_join(tibble(model = names(model_labels)), by = "model") |>
  left_join(pooled_metrics, by = "model") |>
  mutate(
    model_label = unname(model_labels[model]),
    expected_outer_folds = outer_v * outer_repeats,
    failed_outer_folds = expected_outer_folds - replace_na(completed_outer_folds, 0L),
    complete_oof_coverage = replace_na(complete_oof_coverage, FALSE)
  )

best_mean_auc <- max(model_summary$outer_auc_mean, na.rm = TRUE)
model_ranking <- model_summary |>
  mutate(
    within_0.01_of_best_auc = outer_auc_mean >= best_mean_auc - auc_closeness,
    auc_rank = min_rank(desc(outer_auc_mean)),
    calibration_intercept_error = abs(calibration_intercept),
    calibration_slope_error = abs(calibration_slope - 1)
  ) |>
  arrange(
    desc(within_0.01_of_best_auc),
    auc_rank,
    pooled_oof_brier,
    calibration_intercept_error,
    calibration_slope_error
  )

successful_lasso_folds <- n_distinct(raw_results$lasso_summary$outer_id)
selection_frequency <- raw_results$lasso_variables |>
  distinct(outer_id, predictor) |>
  count(predictor, name = "selected_outer_folds") |>
  right_join(tibble(predictor = candidate_predictors), by = "predictor") |>
  mutate(
    selected_outer_folds = replace_na(selected_outer_folds, 0L),
    successful_lasso_folds = .env$successful_lasso_folds,
    selection_frequency = selected_outer_folds / successful_lasso_folds
  ) |>
  arrange(desc(selection_frequency), predictor)

# Audit whether selected numeric hyperparameters repeatedly occur at the
# lower or upper edge of the realized search grid.
numeric_grid_parameters <- names(raw_results$tuning_grids)[
  vapply(raw_results$tuning_grids, is.numeric, logical(1))
]
numeric_grid_parameters <- setdiff(numeric_grid_parameters, "candidate_id")
numeric_grid_parameters <- intersect(
  numeric_grid_parameters, names(raw_results$parameters)
)

grid_ranges <- raw_results$tuning_grids |>
  select(outer_id, model, all_of(numeric_grid_parameters)) |>
  pivot_longer(
    all_of(numeric_grid_parameters),
    names_to = "parameter", values_to = "candidate_value"
  ) |>
  filter(is.finite(candidate_value)) |>
  group_by(outer_id, model, parameter) |>
  summarise(
    grid_min = min(candidate_value),
    grid_max = max(candidate_value),
    .groups = "drop"
  )

best_parameter_long <- raw_results$parameters |>
  select(outer_id, model, all_of(numeric_grid_parameters)) |>
  pivot_longer(
    all_of(numeric_grid_parameters),
    names_to = "parameter", values_to = "best_value"
  ) |>
  filter(is.finite(best_value))

boundary_details <- best_parameter_long |>
  left_join(grid_ranges, by = c("outer_id", "model", "parameter")) |>
  mutate(
    boundary = case_when(
      near(best_value, grid_min) & near(best_value, grid_max) ~ "only_value",
      near(best_value, grid_min) ~ "lower",
      near(best_value, grid_max) ~ "upper",
      TRUE ~ "interior"
    )
  )

boundary_summary <- boundary_details |>
  group_by(model, parameter) |>
  summarise(
    evaluated_outer_folds = n(),
    lower_hits = sum(boundary == "lower"),
    upper_hits = sum(boundary == "upper"),
    boundary_hit_fraction = mean(boundary %in% c("lower", "upper")),
    alert = boundary_hit_fraction > boundary_alert_threshold,
    .groups = "drop"
  ) |>
  arrange(desc(alert), desc(boundary_hit_fraction), model, parameter)

# 7. Outputs --------------------------------------------------
configuration <- tibble(
  setting = c(
    "script_version", "data_file", "data_md5", "seed", "rng_kind",
    "positive_class", "outer_v", "outer_repeats", "inner_v",
    "grid_design", "tuning_selection_metric", "auc_closeness",
    "lasso_alpha", "lasso_primary_rule", "lasso_empty_fallback",
    "boundary_alert_threshold", "parallel_cores", "automatic_final_model"
  ),
  value = as.character(c(
    script_version, data_path, data_md5, seed, paste(RNGkind(), collapse = "; "),
    positive, outer_v, outer_repeats, inner_v,
    grid_design, "Mean inner-CV ROC AUC", auc_closeness,
    1, "lambda.1se", "lambda.min", boundary_alert_threshold,
    raw_results$n_cores, "No - manual review required"
  ))
)

tuning_plan <- tibble(
  model = names(model_labels),
  model_label = unname(model_labels),
  tuning_parameters = unname(model_tuning_parameters),
  planned_candidates = unname(grid_sizes),
  preprocessing = if_else(
    model %in% scale_sensitive_models,
    "Dummy coding, zero-variance removal, normalization",
    "Dummy coding and zero-variance removal"
  )
)

csv_outputs <- list(
  "00_data_audit.csv" = audit,
  "01_lasso_candidate_predictors.csv" = tibble(
    predictor = candidate_predictors,
    status = "Entered into fold-specific LASSO"
  ),
  "02_prespecified_exclusions.csv" = tibble(
    predictor = correlation_drop,
    status = "Prespecified redundancy exclusion"
  ),
  "03_lasso_by_outer_fold.csv" = raw_results$lasso_summary,
  "04_lasso_selected_variables_by_fold.csv" = raw_results$lasso_variables,
  "05_lasso_selection_frequency.csv" = selection_frequency,
  "06_model_tuning_plan.csv" = tuning_plan,
  "07_tuning_candidate_grids.csv" = raw_results$tuning_grids,
  "08_best_parameters_by_fold.csv" = raw_results$parameters,
  "09_hyperparameter_boundary_details.csv" = boundary_details,
  "10_hyperparameter_boundary_summary.csv" = boundary_summary,
  "11_outer_predictions_all_models.csv" = raw_results$predictions,
  "12_patient_level_mean_oof_predictions.csv" = patient_oof,
  "13_prediction_coverage.csv" = prediction_coverage,
  "14_outer_fold_metrics.csv" = fold_metrics,
  "15_model_ranking_FOR_MANUAL_REVIEW.csv" = model_ranking,
  "16_outer_assessment_membership.csv" = raw_results$membership,
  "17_errors.csv" = raw_results$errors,
  "18_tuning_notes.csv" = raw_results$notes,
  "19_run_configuration.csv" = configuration
)

iwalk(csv_outputs, write_table)

# 8. Figures --------------------------------------------------
model_order <- model_ranking$model
auc_mean_points <- model_summary |>
  filter(is.finite(outer_auc_mean)) |>
  mutate(model = factor(model, levels = rev(model_order)))

p_auc <- fold_metrics |>
  mutate(model = factor(model, levels = rev(model_order))) |>
  ggplot(aes(roc_auc, model)) +
  geom_boxplot(
    width = 0.60, outlier.shape = NA,
    fill = "#DCEAF4", color = "#24506A"
  ) +
  geom_point(
    position = position_jitter(width = 0, height = 0.14, seed = seed),
    size = 1.3, alpha = 0.65, color = "#1B6A8F"
  ) +
  geom_point(
    data = auc_mean_points,
    aes(x = outer_auc_mean, y = model),
    inherit.aes = FALSE,
    shape = 18, size = 3.1, color = "#B23A48"
  ) +
  scale_x_continuous(
    limits = c(0.50, 1.00), breaks = seq(0.50, 1.00, 0.10)
  ) +
  scale_y_discrete(labels = function(x) unname(model_labels[x])) +
  labs(
    title = "Nested-CV discrimination across nine models",
    subtitle = paste0(
      "Outer stratified 5-fold CV repeated 5 times; ",
      "diamonds indicate model means"
    ),
    x = "Outer-fold ROC AUC", y = NULL
  ) +
  theme_classic(base_size = 11, base_family = "sans") +
  theme(plot.title = element_text(face = "bold"))

save_plot(p_auc, "Figure_1_outer_AUC_distributions", 8.5, 6.2)

selection_plot_data <- selection_frequency |>
  filter(selected_outer_folds > 0) |>
  mutate(predictor_label = gsub("_", " ", predictor, fixed = TRUE)) |>
  mutate(
    predictor_label = factor(
      predictor_label,
      levels = predictor_label[order(selection_frequency)]
    )
  )

p_selection <- selection_plot_data |>
  ggplot(aes(selection_frequency * 100, predictor_label)) +
  geom_col(fill = "#3B7EA1", width = 0.72) +
  geom_text(
    aes(label = sprintf("%.0f%%", selection_frequency * 100)),
    hjust = -0.10, size = 3
  ) +
  scale_x_continuous(
    limits = c(0, 108), breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"), expand = c(0, 0)
  ) +
  labs(
    title = "LASSO predictor-selection stability",
    subtitle = paste0(
      "Frequency across ", successful_lasso_folds,
      " outer training sets; never-selected predictors are omitted"
    ),
    x = "Outer-fold selection frequency", y = NULL
  ) +
  theme_classic(base_size = 10, base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(size = 8.5),
    plot.margin = margin(5.5, 18, 5.5, 5.5)
  )

selection_height <- max(4.5, 0.34 * nrow(selection_plot_data) + 1.8)
save_plot(
  p_selection, "Figure_2_LASSO_selection_frequency",
  8.5, selection_height
)

# 9. Save complete analysis object ----------------------------
results <- list(
  settings = configuration,
  tuning_plan = tuning_plan,
  audit = audit,
  candidate_predictors = candidate_predictors,
  prespecified_exclusions = correlation_drop,
  outer_folds = raw_results$outer_folds,
  lasso_summary = raw_results$lasso_summary,
  lasso_variables = raw_results$lasso_variables,
  lasso_selection_frequency = selection_frequency,
  tuning_candidate_grids = raw_results$tuning_grids,
  best_parameters = raw_results$parameters,
  hyperparameter_boundary_details = boundary_details,
  hyperparameter_boundary_summary = boundary_summary,
  outer_predictions = raw_results$predictions,
  patient_oof_predictions = patient_oof,
  prediction_coverage = prediction_coverage,
  fold_metrics = fold_metrics,
  model_ranking = model_ranking,
  errors = raw_results$errors,
  tuning_notes = raw_results$notes
)

# 10. Save reproducible analysis results ----------------------

results$session_info <- sessionInfo()

results_path <- file.path(
  out_dir,
  "nested_lasso_9models_results.rds"
)

saveRDS(
  object = results,
  file = results_path,
  compress = "xz",
  version = 3
)

if (!file.exists(results_path)) {
  stop("Failed to save the analysis results.")
}

message(
  "Analysis completed successfully.\n",
  "Results saved to: ",
  normalizePath(results_path, winslash = "/", mustWork = TRUE)
)