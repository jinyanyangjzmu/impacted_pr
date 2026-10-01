# ============================================================
# Impacted stone study
# Fully nested LASSO selection + nine-model comparison
#
# Development cohort only. External-validation data are never read.
#
# Outer loop:
#   unbiased evaluation of the complete selection procedure.
# Inner loop:
#   fold-specific LASSO, preprocessing, hyperparameter tuning,
#   and automatic algorithm selection.
#
# LASSO is calculated once per inner-analysis set and cached for use by
# all nine algorithms. Outer-assessment outcomes are never used for
# feature selection, tuning, or algorithm selection.
#
# Usage:
# Rscript impacted_nested_lasso_9models.R [data.rds] [output_dir]
# ============================================================

# 1. Packages -------------------------------------------------
required_pkgs <- c(
  "tidymodels", "glmnet", "rpart", "ranger", "kernlab", "kknn",
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

# 2. Prespecified settings ------------------------------------
script_version <- "2.0.0"
seed <- 2026L
RNGkind("L'Ecuyer-CMRG")

outer_v <- 5L
outer_repeats <- 5L
# outer_repeats <- 1L
inner_v <- 5L
lasso_v <- 5L
auc_closeness <- 0.01
brier_closeness <- 0.005
calibration_closeness <- 0.10
grid_design <- "max_min_l2"
boundary_alert_threshold <- 0.20

outcome <- "impacted_stone"
id_var <- "patient_id"
positive <- "Impacted"
negative <- "Non-impacted"

# AUC, Brier, and calibration tolerances define practical equivalence.
# This fixed ordering is used only after all three tolerances are satisfied.
complexity_order <- c(
  "logistic", "elastic_net", "decision_tree", "knn", "svm_rbf",
  "random_forest", "xgboost", "lightgbm", "mlp"
)

# More complex search spaces receive larger fixed tuning budgets.
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

# grid_sizes <- c(
#   logistic = 0L,
#   elastic_net = 5L,
#   decision_tree = 5L,
#   random_forest = 5L,
#   svm_rbf = 5L,
#   knn = 5L,
#   lightgbm = 5L,
#   mlp = 5L,
#   xgboost = 5L
# )

scale_sensitive_models <- c("elastic_net", "svm_rbf", "knn", "mlp")

# Outcome-independent, prespecified redundancy exclusions.
correlation_drop <- c(
  "stone_width",
  "hu_above_below_ratio",
  "alcohol_use",
  "hydronephrosis_grade",
  "ureteral_wall_area"
)

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

if (!identical(names(grid_sizes), names(model_labels)) ||
    !identical(names(model_tuning_parameters), names(model_labels)) ||
    !setequal(complexity_order, names(model_labels))) {
  stop("Model settings are not aligned.")
}

# 3. Paths and data -------------------------------------------
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
  file.path(project_dir, "output", "fully_nested_lasso_9models")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, winslash = "/", mustWork = TRUE)

dat <- readRDS(data_path) |> as.data.frame(check.names = FALSE)
required_columns <- c(id_var, outcome, correlation_drop)
if (!all(required_columns %in% names(dat))) {
  stop(
    "Missing required columns: ",
    paste(setdiff(required_columns, names(dat)), collapse = ", ")
  )
}
if (anyNA(dat)) stop("The development dataset contains missing values.")
if (anyDuplicated(dat[[id_var]])) stop(id_var, " must be unique.")

dat[[outcome]] <- factor(
  trimws(as.character(dat[[outcome]])),
  levels = c(positive, negative)
)
if (anyNA(dat[[outcome]])) stop("Unexpected outcome labels.")
if (any(table(dat[[outcome]]) == 0L)) stop("Both outcome classes are required.")

to_factor <- names(dat)[
  vapply(dat, function(x) is.character(x) || is.logical(x), logical(1))
]
to_factor <- setdiff(to_factor, id_var)
dat[to_factor] <- lapply(dat[to_factor], factor)

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

candidate_predictors <- setdiff(
  names(dat), c(id_var, outcome, correlation_drop)
)
if (!length(candidate_predictors)) stop("No candidate predictors were found.")

audit <- tibble(
  item = c(
    "patients", "impacted", "non_impacted", "original_predictors",
    "prespecified_drops", "lasso_candidates", "outer_resamples"
  ),
  value = c(
    nrow(dat), sum(dat[[outcome]] == positive),
    sum(dat[[outcome]] == negative), ncol(dat) - 2L,
    length(correlation_drop), length(candidate_predictors),
    outer_v * outer_repeats
  )
)

# 4. LASSO helpers --------------------------------------------
make_foldid <- function(y, v, seed_value) {
  class_size <- min(table(y))
  v_used <- min(as.integer(v), as.integer(class_size))
  if (v_used < 3L) stop("Too few observations per outcome class for LASSO CV.")

  set.seed(seed_value)
  fold_id <- integer(length(y))
  for (level in sort(unique(y))) {
    index <- which(y == level)
    fold_id[index] <- sample(rep(seq_len(v_used), length.out = length(index)))
  }
  list(fold_id = fold_id, v_used = v_used)
}

select_lasso_variables <- function(
    train_data, predictors, outcome_name, positive_class,
    v, seed_value) {

  formula <- reformulate(predictors, response = outcome_name)
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

  nonzero_variance <- apply(x, 2, stats::sd) > 0
  x <- x[, nonzero_variance, drop = FALSE]
  dummy_map <- dummy_map[nonzero_variance, , drop = FALSE]
  if (!ncol(x)) stop("No non-zero-variance predictors remained for LASSO.")

  y <- as.integer(train_data[[outcome_name]] == positive_class)
  fold <- make_foldid(y, v, seed_value)

  set.seed(seed_value)
  cv_fit <- cv.glmnet(
    x = x,
    y = y,
    family = "binomial",
    alpha = 1,
    foldid = fold$fold_id,
    type.measure = "deviance",
    standardize = TRUE,
    grouped = TRUE
  )

  nonzero_dummies <- function(lambda_value) {
    coefficients <- as.matrix(stats::coef(cv_fit, s = lambda_value))
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
  selected <- intersect(predictors, selected)

  list(
    selected = selected,
    selected_dummies = selected_dummies,
    lambda_rule = lambda_rule,
    lambda_used = as.numeric(lambda_used),
    lambda_1se = as.numeric(cv_fit$lambda.1se),
    lambda_min = as.numeric(cv_fit$lambda.min),
    n_selected = length(selected),
    n_selected_dummies = length(selected_dummies),
    lasso_v = fold$v_used
  )
}

make_split_key <- function(ids) {
  paste(sort(as.character(ids)), collapse = "\u001f")
}

count_design_columns <- function(data, predictors) {
  x <- model.matrix(reformulate(predictors), data)
  x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  if (ncol(x)) x <- x[, apply(x, 2, stats::sd) > 0, drop = FALSE]
  max(1L, ncol(x))
}

# 5. Cached supervised-selection recipe step -----------------
# The cache is built solely from each analysis set before tuning. During
# recipe preparation, the patient IDs identify the matching analysis set.
step_cached_lasso_select <- function(
    recipe, predictors, outcome_name, id_column, selection_cache,
    role = NA, trained = FALSE, selected_predictors = NULL,
    skip = FALSE, id = recipes::rand_id("cached_lasso_select")) {

  recipes::add_step(
    recipe,
    step_cached_lasso_select_new(
      predictors = predictors,
      outcome_name = outcome_name,
      id_column = id_column,
      selection_cache = selection_cache,
      role = role,
      trained = trained,
      selected_predictors = selected_predictors,
      skip = skip,
      id = id
    )
  )
}

step_cached_lasso_select_new <- function(
    predictors, outcome_name, id_column, selection_cache,
    role, trained, selected_predictors, skip, id) {

  recipes::step(
    subclass = "cached_lasso_select",
    predictors = predictors,
    outcome_name = outcome_name,
    id_column = id_column,
    selection_cache = selection_cache,
    role = role,
    trained = trained,
    selected_predictors = selected_predictors,
    skip = skip,
    id = id
  )
}

prep.step_cached_lasso_select <- function(x, training, info = NULL, ...) {
  if (!x$id_column %in% names(training)) {
    stop("ID column is missing during cached LASSO recipe preparation.")
  }
  key <- make_split_key(training[[x$id_column]])
  selected <- x$selection_cache[[key]]
  if (is.null(selected)) {
    stop("No cached LASSO result matched the current analysis set.")
  }
  if (!length(selected) || !all(selected %in% names(training))) {
    stop("Cached LASSO predictors are empty or absent from the analysis set.")
  }

  step_cached_lasso_select_new(
    predictors = x$predictors,
    outcome_name = x$outcome_name,
    id_column = x$id_column,
    selection_cache = x$selection_cache,
    role = x$role,
    trained = TRUE,
    selected_predictors = selected,
    skip = x$skip,
    id = x$id
  )
}

bake.step_cached_lasso_select <- function(object, new_data, ...) {
  remove <- setdiff(object$predictors, object$selected_predictors)
  new_data[, setdiff(names(new_data), remove), drop = FALSE]
}

print.step_cached_lasso_select <- function(x, width = max(20, options()$width - 30), ...) {
  cat("Cached fold-specific LASSO selection for ", length(x$predictors),
      " candidate predictors\n", sep = "")
  invisible(x)
}

tidy.step_cached_lasso_select <- function(x, ...) {
  terms <- if (isTRUE(x$trained)) x$selected_predictors else x$predictors
  tibble(terms = terms, retained = terms %in% x$selected_predictors, id = x$id)
}

required_pkgs.step_cached_lasso_select <- function(x, ...) character(0)

# 6. Recipes and model definitions ----------------------------
make_recipe <- function(
    train_data, predictors, selection_cache, normalize = FALSE) {

  rec <- recipe(
    reformulate(c(id_var, predictors), response = outcome),
    data = train_data
  ) |>
    update_role(all_of(id_var), new_role = "id") |>
    step_cached_lasso_select(
      predictors = predictors,
      outcome_name = outcome,
      id_column = id_var,
      selection_cache = selection_cache
    ) |>
    step_unknown(all_nominal_predictors(), new_level = "unknown") |>
    step_novel(all_nominal_predictors(), new_level = "novel") |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors())

  if (normalize) rec <- rec |> step_normalize(all_numeric_predictors())
  rec
}

make_models <- function(p_mtry) {
  p_mtry <- max(1L, as.integer(p_mtry))
  mtry_value <- if (p_mtry > 1L) tune() else 1L
  mtry_parameter <- if (p_mtry > 1L) {
    list(dials::mtry(c(1L, p_mtry)))
  } else {
    list()
  }
  parameter_set <- function(...) do.call(dials::parameters, list(...))

  boost_definition <- function(engine, min_n_range) {
    spec <- boost_tree(
      mtry = !!mtry_value,
      trees = tune(),
      min_n = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      loss_reduction = tune(),
      sample_size = tune()
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

# 7. Metrics and selection rule -------------------------------
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
    unname(stats::coef(glm(y ~ offset(lp), family = binomial()))[1]),
    error = function(e) NA_real_
  )
  slope <- tryCatch(
    unname(stats::coef(glm(y ~ lp, family = binomial()))[2]),
    error = function(e) NA_real_
  )
  tibble(calibration_intercept = intercept, calibration_slope = slope)
}

choose_model <- function(selection_input, context_id) {
  required <- c(
    "model", "selection_auc", "selection_brier",
    "calibration_intercept", "calibration_slope", "complete"
  )
  if (!all(required %in% names(selection_input))) {
    stop("Model-selection input is missing required columns.")
  }

  ranked <- selection_input |>
    mutate(
      complexity_rank = match(model, complexity_order),
      calibration_error = abs(calibration_intercept) +
        abs(calibration_slope - 1),
      valid = complete & is.finite(selection_auc) &
        is.finite(selection_brier) & is.finite(calibration_error)
    )
  if (!any(ranked$valid)) stop("No complete model was available for selection.")

  best_auc <- max(ranked$selection_auc[ranked$valid])
  ranked <- ranked |>
    mutate(
      within_auc_window = valid & selection_auc >= best_auc - auc_closeness
    )
  best_brier <- min(
    ranked$selection_brier[ranked$within_auc_window], na.rm = TRUE
  )
  ranked <- ranked |>
    mutate(
      within_brier_window = within_auc_window &
        selection_brier <= best_brier + brier_closeness
    )
  best_calibration_error <- min(
    ranked$calibration_error[ranked$within_brier_window], na.rm = TRUE
  )
  ranked <- ranked |>
    mutate(
      within_calibration_window = within_brier_window &
        calibration_error <= best_calibration_error + calibration_closeness
    ) |>
    arrange(
      desc(within_calibration_window),
      desc(within_brier_window),
      desc(within_auc_window),
      selection_brier,
      calibration_error,
      complexity_rank
    )

  selected <- ranked |>
    filter(within_calibration_window) |>
    arrange(complexity_rank) |>
    slice_head(n = 1L) |>
    pull(model)

  ranked <- ranked |>
    mutate(
      selected = model == selected,
      selection_context = context_id,
      best_auc = best_auc,
      auc_closeness = auc_closeness,
      best_brier = best_brier,
      brier_closeness = brier_closeness,
      best_calibration_error = best_calibration_error,
      calibration_closeness = calibration_closeness
    )

  list(selected_model = selected, ranking = ranked)
}

selection_rule <- tibble(
  stage = c(rep("Within each outer training set", 5L), "Final aggregation"),
  priority = c(1:5, 1L),
  rule = c(
    "Exclude incomplete or non-finite models",
    "Retain models with mean ROC AUC within 0.01 of the best",
    "Retain models with mean Brier score within 0.005 of the best",
    paste(
      "Retain models whose |intercept| + |slope - 1| is within 0.10",
      "of the best"
    ),
    "Among practically equivalent models, use the fixed complexity order",
    paste(
      "Choose the algorithm selected in the most outer folds;",
      "resolve a frequency tie using the fixed complexity order"
    )
  )
)

# 8. Inner tuning and outer fitting ---------------------------
run_inner_model <- function(
    model_name, definition, model_recipe, train_outer, inner_folds,
    outer_id, seed_value) {

  workflow_template <- workflow() |>
    add_recipe(model_recipe) |>
    add_model(definition$spec)

  tuning_grid <- tibble()
  tuning_notes <- tibble()

  if (is.null(definition$params)) {
    set.seed(seed_value)
    resampled <- fit_resamples(
      workflow_template,
      resamples = inner_folds,
      metrics = metric_set(roc_auc, brier_class),
      control = control_resamples(
        save_pred = TRUE,
        save_workflow = FALSE,
        verbose = FALSE,
        allow_par = TRUE,
        parallel_over = "resamples",
        pkgs = required_pkgs
      )
    )
    best_parameters <- tibble(.config = "Preprocessor1_Model1")
    metric_table <- collect_metrics(resampled)
    prediction_table <- collect_predictions(resampled)
    tuning_notes <- collect_notes(resampled)
  } else {
    set.seed(seed_value)
    grid <- dials::grid_space_filling(
      definition$params,
      size = unname(grid_sizes[[model_name]]),
      type = grid_design
    )
    tuning_grid <- grid |>
      mutate(
        outer_id = outer_id,
        model = model_name,
        candidate_id = row_number(),
        .before = 1
      )

    set.seed(seed_value)
    resampled <- tune_grid(
      workflow_template,
      resamples = inner_folds,
      grid = grid,
      metrics = metric_set(roc_auc, brier_class),
      control = control_grid(
        save_pred = TRUE,
        save_workflow = FALSE,
        verbose = FALSE,
        allow_par = TRUE,
        parallel_over = "resamples",
        pkgs = required_pkgs
      )
    )
    metric_table <- collect_metrics(resampled)
    if (!any(metric_table$.metric == "roc_auc" & is.finite(metric_table$mean))) {
      stop("No valid inner-CV ROC AUC estimates for ", model_name, ".")
    }
    best_parameters <- select_best(resampled, metric = "roc_auc")
    best_config <- best_parameters$.config[[1]]
    prediction_table <- collect_predictions(resampled) |>
      filter(.config == best_config)
    tuning_notes <- collect_notes(resampled)
  }

  best_config <- best_parameters$.config[[1]]
  best_metrics <- metric_table |>
    filter(.config == best_config)
  inner_auc_mean <- best_metrics |>
    filter(.metric == "roc_auc") |>
    pull(mean)
  inner_brier_mean <- best_metrics |>
    filter(.metric == "brier_class") |>
    pull(mean)

  if (length(inner_auc_mean) != 1L || length(inner_brier_mean) != 1L) {
    stop("Best inner metrics were missing or duplicated for ", model_name, ".")
  }

  probability_column <- paste0(".pred_", make.names(positive))
  if (!all(c(".row", outcome, probability_column) %in% names(prediction_table))) {
    stop("Inner predictions are missing required columns for ", model_name, ".")
  }

  row_key <- tibble(
    .row = seq_len(nrow(train_outer)),
    patient_id = train_outer[[id_var]]
  )
  inner_predictions <- prediction_table |>
    left_join(row_key, by = ".row") |>
    transmute(
      outer_id = outer_id,
      model = model_name,
      patient_id = patient_id,
      truth = .data[[outcome]],
      probability = .data[[probability_column]],
      inner_resample = if ("id2" %in% names(prediction_table)) {
        paste(.data$id, .data$id2, sep = "_")
      } else {
        as.character(.data$id)
      }
    )

  complete_coverage <- nrow(inner_predictions) == nrow(train_outer) &&
    n_distinct(inner_predictions$patient_id) == nrow(train_outer) &&
    !anyDuplicated(inner_predictions$patient_id)
  calibration <- calibration_metrics(
    inner_predictions$truth,
    inner_predictions$probability
  )

  summary_row <- tibble(
    outer_id = outer_id,
    model = model_name,
    inner_auc_mean = inner_auc_mean,
    inner_brier_mean = inner_brier_mean,
    inner_oof_auc = safe_auc(
      inner_predictions$truth,
      inner_predictions$probability
    ),
    inner_oof_brier = mean(
      (as.integer(inner_predictions$truth == positive) -
         inner_predictions$probability)^2
    ),
    calibration_intercept = calibration$calibration_intercept,
    calibration_slope = calibration$calibration_slope,
    covered_patients = n_distinct(inner_predictions$patient_id),
    expected_patients = nrow(train_outer),
    complete_inner_coverage = complete_coverage,
    best_config = best_config
  )

  parameter_row <- best_parameters |>
    mutate(outer_id = outer_id, model = model_name, .before = 1)
  metric_table <- metric_table |>
    mutate(outer_id = outer_id, model = model_name, .before = 1)
  if (nrow(tuning_notes)) {
    tuning_notes <- tuning_notes |>
      mutate(outer_id = outer_id, model = model_name, .before = 1)
  }

  list(
    summary = summary_row,
    best_parameters = parameter_row,
    tuning_grid = tuning_grid,
    tuning_metrics = metric_table,
    predictions = inner_predictions,
    notes = tuning_notes,
    workflow_template = workflow_template
  )
}

adjust_mtry <- function(best_parameters, p_outer) {
  adjusted <- best_parameters
  if ("mtry" %in% names(adjusted)) {
    adjusted$mtry <- pmax(1L, pmin(as.integer(round(adjusted$mtry)), p_outer))
  }
  adjusted
}

fit_outer_model <- function(
    model_name, workflow_template, best_parameters, train_outer,
    assess_outer, p_outer, outer_id, seed_value) {

  parameters_for_fit <- adjust_mtry(best_parameters, p_outer)
  if (ncol(parameters_for_fit) == 1L &&
      identical(names(parameters_for_fit), ".config")) {
    final_workflow <- workflow_template
  } else {
    final_workflow <- finalize_workflow(workflow_template, parameters_for_fit)
  }

  set.seed(seed_value)
  fitted <- fit(final_workflow, data = train_outer)
  probability_column <- paste0(".pred_", make.names(positive))
  probability <- predict(fitted, assess_outer, type = "prob")
  if (!probability_column %in% names(probability)) {
    stop("Positive-class probability column was not found.")
  }
  probability_value <- probability[[probability_column]]
  if (any(!is.finite(probability_value)) ||
      any(probability_value < 0 | probability_value > 1)) {
    stop("Model returned invalid outer-assessment probabilities.")
  }

  list(
    predictions = tibble(
      patient_id = assess_outer[[id_var]],
      truth = assess_outer[[outcome]],
      probability = probability_value,
      outer_id = outer_id,
      model = model_name
    ),
    parameters_used = parameters_for_fit |>
      mutate(outer_id = outer_id, model = model_name, .before = 1)
  )
}

# 9. Fully nested analysis ------------------------------------
run_analysis <- function() {
  available_cores <- parallel::detectCores(logical = FALSE)
  if (is.na(available_cores)) available_cores <- 2L
  n_cores <- max(1L, min(8L, available_cores - 1L))

  cluster <- parallel::makePSOCKcluster(n_cores)
  parallel::clusterExport(
    cluster,
    varlist = c(
      "make_split_key",
      "step_cached_lasso_select_new",
      "prep.step_cached_lasso_select",
      "bake.step_cached_lasso_select",
      "print.step_cached_lasso_select",
      "tidy.step_cached_lasso_select",
      "required_pkgs.step_cached_lasso_select"
    ),
    envir = globalenv()
  )
  doParallel::registerDoParallel(cluster)
  on.exit({
    parallel::stopCluster(cluster)
    foreach::registerDoSEQ()
  }, add = TRUE)

  set.seed(seed)
  outer_folds <- vfold_cv(
    dat, v = outer_v, repeats = outer_repeats, strata = impacted_stone
  )

  log <- list(
    inner_lasso_summary = list(),
    inner_lasso_variables = list(),
    outer_lasso_summary = list(),
    outer_lasso_variables = list(),
    inner_model_summary = list(),
    inner_best_parameters = list(),
    tuning_grids = list(),
    tuning_metrics = list(),
    inner_predictions = list(),
    tuning_notes = list(),
    outer_choices = list(),
    outer_predictions_all = list(),
    outer_predictions_selected = list(),
    outer_parameters_used = list(),
    membership = list(),
    errors = list()
  )

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
    log$membership[[outer_id]] <- tibble(
      outer_id = outer_id,
      patient_id = assess_outer[[id_var]],
      role = "assessment"
    )

    if (any(table(train_outer[[outcome]]) < inner_v)) {
      stop("An outcome class has fewer observations than inner_v in ", outer_id)
    }
    set.seed(seed + 10000L + i)
    inner_folds <- vfold_cv(
      train_outer, v = inner_v, strata = impacted_stone
    )

    # Precompute LASSO exactly once for every inner-analysis set.
    selection_cache <- list()
    inner_design_columns <- integer(nrow(inner_folds))
    for (k in seq_len(nrow(inner_folds))) {
      inner_id <- paste0(outer_id, "_Inner", k)
      inner_train <- analysis(inner_folds$splits[[k]])
      lasso_seed <- seed + 200000L + i * 100L + k
      lasso_inner <- select_lasso_variables(
        inner_train,
        candidate_predictors,
        outcome,
        positive,
        lasso_v,
        lasso_seed
      )
      key <- make_split_key(inner_train[[id_var]])
      if (!is.null(selection_cache[[key]])) {
        stop("Duplicated inner-analysis cache key in ", outer_id)
      }
      selection_cache[[key]] <- lasso_inner$selected
      inner_design_columns[[k]] <- count_design_columns(
        inner_train, lasso_inner$selected
      )

      log$inner_lasso_summary[[inner_id]] <- tibble(
        outer_id = outer_id,
        inner_id = inner_id,
        n_analysis = nrow(inner_train),
        lasso_v = lasso_inner$lasso_v,
        lambda_rule = lasso_inner$lambda_rule,
        lambda_used = lasso_inner$lambda_used,
        lambda_1se = lasso_inner$lambda_1se,
        lambda_min = lasso_inner$lambda_min,
        n_selected = lasso_inner$n_selected,
        n_selected_dummies = lasso_inner$n_selected_dummies,
        design_columns = inner_design_columns[[k]]
      )
      log$inner_lasso_variables[[inner_id]] <- tibble(
        outer_id = outer_id,
        inner_id = inner_id,
        predictor = lasso_inner$selected
      )
    }

    # Cache the complete outer-training LASSO for the later outer refit.
    outer_lasso_seed <- seed + 300000L + i
    lasso_outer <- select_lasso_variables(
      train_outer,
      candidate_predictors,
      outcome,
      positive,
      lasso_v,
      outer_lasso_seed
    )
    outer_key <- make_split_key(train_outer[[id_var]])
    selection_cache[[outer_key]] <- lasso_outer$selected
    p_outer <- count_design_columns(train_outer, lasso_outer$selected)
    p_mtry <- min(inner_design_columns)

    log$outer_lasso_summary[[outer_id]] <- tibble(
      outer_id = outer_id,
      n_analysis = nrow(train_outer),
      lasso_v = lasso_outer$lasso_v,
      lambda_rule = lasso_outer$lambda_rule,
      lambda_used = lasso_outer$lambda_used,
      lambda_1se = lasso_outer$lambda_1se,
      lambda_min = lasso_outer$lambda_min,
      n_selected = lasso_outer$n_selected,
      n_selected_dummies = lasso_outer$n_selected_dummies,
      design_columns = p_outer,
      tuning_mtry_upper_bound = p_mtry
    )
    log$outer_lasso_variables[[outer_id]] <- tibble(
      outer_id = outer_id,
      predictor = lasso_outer$selected
    )

    definitions <- make_models(p_mtry)
    model_results <- list()

    for (j in seq_along(definitions)) {
      model_name <- names(definitions)[[j]]
      message("  Inner evaluation: ", model_labels[[model_name]])
      model_recipe <- make_recipe(
        train_outer,
        candidate_predictors,
        selection_cache,
        normalize = model_name %in% scale_sensitive_models
      )

      result <- tryCatch(
        run_inner_model(
          model_name = model_name,
          definition = definitions[[model_name]],
          model_recipe = model_recipe,
          train_outer = train_outer,
          inner_folds = inner_folds,
          outer_id = outer_id,
          seed_value = seed + 400000L + i * 100L + j
        ),
        error = function(e) e
      )

      key <- paste(outer_id, model_name, sep = "_")
      if (inherits(result, "error")) {
        message("    FAILED: ", conditionMessage(result))
        log$errors[[key]] <- tibble(
          outer_id = outer_id,
          model = model_name,
          stage = "inner_tuning",
          error = conditionMessage(result)
        )
      } else {
        model_results[[model_name]] <- result
        log$inner_model_summary[[key]] <- result$summary
        log$inner_best_parameters[[key]] <- result$best_parameters
        if (nrow(result$tuning_grid)) {
          log$tuning_grids[[key]] <- result$tuning_grid
        }
        log$tuning_metrics[[key]] <- result$tuning_metrics
        log$inner_predictions[[key]] <- result$predictions
        if (nrow(result$notes)) log$tuning_notes[[key]] <- result$notes
      }
    }

    inner_summary_outer <- bind_rows(
      lapply(model_results, function(x) x$summary)
    )
    selection_input <- inner_summary_outer |>
      transmute(
        model,
        selection_auc = inner_auc_mean,
        selection_brier = inner_brier_mean,
        calibration_intercept,
        calibration_slope,
        complete = complete_inner_coverage
      ) |>
      right_join(tibble(model = names(model_labels)), by = "model") |>
      mutate(complete = replace_na(complete, FALSE))
    choice <- choose_model(selection_input, context_id = outer_id)
    selected_model_outer <- choice$selected_model
    log$outer_choices[[outer_id]] <- choice$ranking |>
      mutate(outer_id = outer_id, .before = 1)

    # Fit all nine inner-optimized algorithms on the complete outer-training
    # data for secondary paired comparison. Only the algorithm selected from
    # inner results contributes to the primary selected-pipeline prediction.
    for (j in seq_along(model_results)) {
      model_name <- names(model_results)[[j]]
      message("  Outer fit: ", model_labels[[model_name]])
      inner_result <- model_results[[model_name]]
      parameter_values <- inner_result$best_parameters |>
        select(-outer_id, -model)

      outer_fit <- tryCatch(
        fit_outer_model(
          model_name = model_name,
          workflow_template = inner_result$workflow_template,
          best_parameters = parameter_values,
          train_outer = train_outer,
          assess_outer = assess_outer,
          p_outer = p_outer,
          outer_id = outer_id,
          seed_value = seed + 500000L + i * 100L + j
        ),
        error = function(e) e
      )

      key <- paste0(outer_id, "_", model_name, "_outer")
      if (inherits(outer_fit, "error")) {
        log$errors[[key]] <- tibble(
          outer_id = outer_id,
          model = model_name,
          stage = "outer_refit_or_prediction",
          error = conditionMessage(outer_fit)
        )
        if (identical(model_name, selected_model_outer)) {
          stop(
            "The inner-selected model failed during outer evaluation in ",
            outer_id, ": ", conditionMessage(outer_fit)
          )
        }
      } else {
        log$outer_predictions_all[[key]] <- outer_fit$predictions
        log$outer_parameters_used[[key]] <- outer_fit$parameters_used
        if (identical(model_name, selected_model_outer)) {
          log$outer_predictions_selected[[outer_id]] <-
            outer_fit$predictions |>
            mutate(selected_model = model_name)
        }
      }
    }

    checkpoint <- list(
      completed_outer_resamples = i,
      inner_lasso_summary = bind_rows(log$inner_lasso_summary),
      outer_lasso_summary = bind_rows(log$outer_lasso_summary),
      inner_model_summary = bind_rows(log$inner_model_summary),
      outer_choices = bind_rows(log$outer_choices),
      outer_predictions_all = bind_rows(log$outer_predictions_all),
      outer_predictions_selected = bind_rows(log$outer_predictions_selected),
      errors = bind_rows(log$errors)
    )
    saveRDS(
      checkpoint,
      file.path(out_dir, "checkpoint_fully_nested_cv.rds"),
      compress = "xz",
      version = 3
    )
  }

  c(
    list(outer_folds = outer_folds, n_cores = n_cores),
    lapply(log, bind_rows)
  )
}

raw_results <- run_analysis()
if (!nrow(raw_results$outer_predictions_selected)) {
  stop("No selected-pipeline outer predictions were produced.")
}

# 10. Performance summaries ----------------------------------
all_model_fold_metrics <- raw_results$outer_predictions_all |>
  group_by(outer_id, model) |>
  summarise(
    n = n(),
    roc_auc = safe_auc(truth, probability),
    brier = mean((as.integer(truth == positive) - probability)^2),
    .groups = "drop"
  )

all_model_patient_oof <- raw_results$outer_predictions_all |>
  group_by(patient_id, model, truth) |>
  summarise(
    probability = mean(probability),
    n_predictions = n(),
    .groups = "drop"
  )

all_model_pooled <- all_model_patient_oof |>
  group_by(model) |>
  group_modify(~ {
    complete <- nrow(.x) == nrow(dat) &&
      all(.x$n_predictions == outer_repeats)
    calibration <- if (complete) {
      calibration_metrics(.x$truth, .x$probability)
    } else {
      tibble(calibration_intercept = NA_real_, calibration_slope = NA_real_)
    }
    tibble(
      pooled_oof_auc = if (complete) safe_auc(.x$truth, .x$probability) else NA_real_,
      pooled_oof_brier = if (complete) {
        mean((as.integer(.x$truth == positive) - .x$probability)^2)
      } else {
        NA_real_
      },
      calibration_intercept = calibration$calibration_intercept,
      calibration_slope = calibration$calibration_slope,
      n_patients = nrow(.x),
      complete_oof_coverage = complete
    )
  }) |>
  ungroup()

all_model_summary <- all_model_fold_metrics |>
  group_by(model) |>
  summarise(
    completed_outer_folds = sum(is.finite(roc_auc)),
    outer_auc_mean = mean(roc_auc, na.rm = TRUE),
    outer_auc_sd = stats::sd(roc_auc, na.rm = TRUE),
    outer_brier_mean = mean(brier, na.rm = TRUE),
    .groups = "drop"
  ) |>
  right_join(tibble(model = names(model_labels)), by = "model") |>
  left_join(all_model_pooled, by = "model") |>
  mutate(
    model_label = unname(model_labels[model]),
    expected_outer_folds = outer_v * outer_repeats,
    completed_outer_folds = replace_na(completed_outer_folds, 0L),
    failed_outer_folds = expected_outer_folds - completed_outer_folds,
    complete_oof_coverage = replace_na(complete_oof_coverage, FALSE)
  )

# The final algorithm is aggregated from inner-only choices. Outer-assessment
# performance is retained for secondary paired comparison but never determines
# selected_model. This preserves the independence of every outer assessment.
outer_choice_frequency <- raw_results$outer_choices |>
  filter(selected) |>
  count(model, name = "selected_outer_folds") |>
  right_join(tibble(model = names(model_labels)), by = "model") |>
  mutate(
    selected_outer_folds = replace_na(selected_outer_folds, 0L),
    selection_frequency = selected_outer_folds / (outer_v * outer_repeats),
    model_label = unname(model_labels[model]),
    complexity_rank = match(model, complexity_order)
  ) |>
  arrange(desc(selected_outer_folds), complexity_rank)

final_choice_candidates <- outer_choice_frequency |>
  left_join(
    all_model_summary |>
      select(model, failed_outer_folds, complete_oof_coverage),
    by = "model"
  ) |>
  mutate(
    valid_for_finalization = failed_outer_folds == 0L & complete_oof_coverage
  )
if (!any(final_choice_candidates$valid_for_finalization)) {
  stop("No algorithm was eligible for finalization.")
}
selected_model <- final_choice_candidates |>
  filter(valid_for_finalization) |>
  arrange(desc(selected_outer_folds), complexity_rank) |>
  slice_head(n = 1L) |>
  pull(model)

model_ranking <- outer_choice_frequency |>
  left_join(all_model_summary, by = c("model", "model_label")) |>
  mutate(
    valid_for_finalization = failed_outer_folds == 0L & complete_oof_coverage,
    selected = model == selected_model,
    selection_source = "Frequency of inner-only choices across outer folds"
  ) |>
  arrange(desc(selected), desc(selected_outer_folds), complexity_rank)

selected_fold_metrics <- raw_results$outer_predictions_selected |>
  group_by(outer_id, selected_model) |>
  summarise(
    n = n(),
    roc_auc = safe_auc(truth, probability),
    brier = mean((as.integer(truth == positive) - probability)^2),
    .groups = "drop"
  )

selected_patient_oof <- raw_results$outer_predictions_selected |>
  group_by(patient_id, truth) |>
  summarise(
    probability = mean(probability),
    n_predictions = n(),
    selected_models = paste(sort(unique(selected_model)), collapse = ";"),
    .groups = "drop"
  )

selected_complete <- nrow(selected_patient_oof) == nrow(dat) &&
  all(selected_patient_oof$n_predictions == outer_repeats)
if (!selected_complete) {
  stop("Selected-pipeline outer prediction coverage is incomplete.")
}
selected_calibration <- calibration_metrics(
  selected_patient_oof$truth,
  selected_patient_oof$probability
)
selected_pipeline_performance <- tibble(
  completed_outer_folds = nrow(selected_fold_metrics),
  expected_outer_folds = outer_v * outer_repeats,
  outer_auc_mean = mean(selected_fold_metrics$roc_auc),
  outer_auc_sd = stats::sd(selected_fold_metrics$roc_auc),
  pooled_oof_auc = safe_auc(
    selected_patient_oof$truth,
    selected_patient_oof$probability
  ),
  pooled_oof_brier = mean(
    (as.integer(selected_patient_oof$truth == positive) -
       selected_patient_oof$probability)^2
  ),
  calibration_intercept = selected_calibration$calibration_intercept,
  calibration_slope = selected_calibration$calibration_slope,
  n_patients = nrow(selected_patient_oof),
  complete_oof_coverage = selected_complete
)

successful_outer_lasso <- n_distinct(raw_results$outer_lasso_summary$outer_id)
lasso_selection_frequency <- raw_results$outer_lasso_variables |>
  distinct(outer_id, predictor) |>
  count(predictor, name = "selected_outer_folds") |>
  right_join(tibble(predictor = candidate_predictors), by = "predictor") |>
  mutate(
    selected_outer_folds = replace_na(selected_outer_folds, 0L),
    successful_lasso_folds = successful_outer_lasso,
    selection_frequency = selected_outer_folds / successful_outer_lasso
  ) |>
  arrange(desc(selection_frequency), predictor)

# 11. Hyperparameter-boundary audit ---------------------------
numeric_grid_parameters <- names(raw_results$tuning_grids)[
  vapply(raw_results$tuning_grids, is.numeric, logical(1))
]
numeric_grid_parameters <- setdiff(
  numeric_grid_parameters,
  c("candidate_id", "outer_id")
)
numeric_grid_parameters <- intersect(
  numeric_grid_parameters,
  names(raw_results$inner_best_parameters)
)

if (length(numeric_grid_parameters)) {
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

  best_parameter_long <- raw_results$inner_best_parameters |>
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
} else {
  boundary_details <- tibble()
  boundary_summary <- tibble()
}

# 12. Outputs -------------------------------------------------
configuration <- tibble(
  setting = c(
    "script_version", "data_file", "data_md5", "seed", "rng_kind",
    "positive_class", "outer_v", "outer_repeats", "inner_v", "lasso_v",
    "grid_design", "tuning_selection_metric", "algorithm_auc_closeness",
    "algorithm_brier_closeness", "algorithm_calibration_closeness",
    "algorithm_secondary_metrics", "algorithm_final_tie_break",
    "lasso_alpha", "lasso_primary_rule", "lasso_empty_fallback",
    "inner_lasso_cache", "automatic_final_model",
    "final_model_aggregation_rule", "selected_model",
    "parallel_cores", "external_data_accessed"
  ),
  value = as.character(c(
    script_version, data_path, data_md5, seed, paste(RNGkind(), collapse = "; "),
    positive, outer_v, outer_repeats, inner_v, lasso_v,
    grid_design, "Mean inner-CV ROC AUC", auc_closeness,
    brier_closeness, calibration_closeness,
    "Mean Brier; |calibration intercept| + |calibration slope - 1|",
    "Fixed complexity order",
    1, "lambda.1se", "lambda.min",
    "One LASSO per inner-analysis set, shared by all algorithms",
    "Yes", "Most inner-selected outer folds; fixed complexity tie-break",
    selected_model, raw_results$n_cores, "No"
  ))
)

tuning_plan <- tibble(
  model = names(model_labels),
  model_label = unname(model_labels),
  tuning_parameters = unname(model_tuning_parameters),
  planned_candidates = unname(grid_sizes),
  preprocessing = if_else(
    model %in% scale_sensitive_models,
    "Fold-specific LASSO; dummy coding; zero-variance removal; normalization",
    "Fold-specific LASSO; dummy coding; zero-variance removal"
  )
)

write_table <- function(data, filename) {
  if (!ncol(data) || !nrow(data)) data <- tibble(message = "No records")
  write_excel_csv(data, file.path(out_dir, filename))
}

csv_outputs <- list(
  "00_data_audit.csv" = audit,
  "01_lasso_candidate_predictors.csv" = tibble(
    predictor = candidate_predictors,
    status = "Entered into every fold-specific LASSO"
  ),
  "02_prespecified_exclusions.csv" = tibble(
    predictor = correlation_drop,
    status = "Prespecified outcome-independent redundancy exclusion"
  ),
  "03_inner_lasso_by_fold.csv" = raw_results$inner_lasso_summary,
  "04_inner_lasso_variables_by_fold.csv" = raw_results$inner_lasso_variables,
  "05_outer_lasso_by_fold.csv" = raw_results$outer_lasso_summary,
  "06_outer_lasso_variables_by_fold.csv" = raw_results$outer_lasso_variables,
  "07_outer_lasso_selection_frequency.csv" = lasso_selection_frequency,
  "08_model_tuning_plan.csv" = tuning_plan,
  "09_inner_model_summary_by_outer_fold.csv" = raw_results$inner_model_summary,
  "10_inner_best_parameters_by_outer_fold.csv" = raw_results$inner_best_parameters,
  "11_tuning_candidate_grids.csv" = raw_results$tuning_grids,
  "12_tuning_metrics.csv" = raw_results$tuning_metrics,
  "13_algorithm_choice_by_outer_fold.csv" = raw_results$outer_choices,
  "14_algorithm_selection_frequency.csv" = outer_choice_frequency,
  "15_outer_predictions_all_models.csv" = raw_results$outer_predictions_all,
  "16_outer_predictions_selected_pipeline.csv" = raw_results$outer_predictions_selected,
  "17_all_model_outer_fold_metrics.csv" = all_model_fold_metrics,
  "18_all_model_performance_summary.csv" = all_model_summary,
  "19_model_ranking_and_final_selection.csv" = model_ranking,
  "20_selected_pipeline_outer_fold_metrics.csv" = selected_fold_metrics,
  "21_selected_pipeline_patient_oof.csv" = selected_patient_oof,
  "22_selected_pipeline_performance.csv" = selected_pipeline_performance,
  "23_outer_parameters_actually_used.csv" = raw_results$outer_parameters_used,
  "24_hyperparameter_boundary_details.csv" = boundary_details,
  "25_hyperparameter_boundary_summary.csv" = boundary_summary,
  "26_model_selection_rule.csv" = selection_rule,
  "27_outer_assessment_membership.csv" = raw_results$membership,
  "28_errors.csv" = raw_results$errors,
  "29_tuning_notes.csv" = raw_results$tuning_notes,
  "30_run_configuration.csv" = configuration
)
iwalk(csv_outputs, write_table)

# 13. Diagnostic figures -------------------------------------
save_plot <- function(plot, filename, width, height) {
  pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
  ggsave(
    file.path(out_dir, paste0(filename, ".pdf")),
    plot, width = width, height = height, device = pdf_device
  )
  ggsave(
    file.path(out_dir, paste0(filename, ".png")),
    plot, width = width, height = height, dpi = 600, bg = "white"
  )
}

model_order <- model_ranking$model
p_auc <- all_model_fold_metrics |>
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
  scale_x_continuous(limits = c(0.50, 1.00), breaks = seq(0.50, 1.00, 0.10)) +
  scale_y_discrete(labels = function(x) unname(model_labels[x])) +
  labs(
    title = "Outer-CV discrimination of fixed candidate algorithms",
    subtitle = paste0(
      "Fold-specific LASSO and inner tuning; final algorithm selection is separate"
    ),
    x = "Outer-fold ROC AUC", y = NULL
  ) +
  theme_classic(base_size = 11, base_family = "sans") +
  theme(plot.title = element_text(face = "bold"))
save_plot(p_auc, "Figure_1_all_model_outer_AUC", 8.5, 6.2)

p_choice <- outer_choice_frequency |>
  mutate(
    model_label = factor(
      model_label,
      levels = rev(model_label[order(selected_outer_folds, model_label)])
    )
  ) |>
  ggplot(aes(selected_outer_folds, model_label)) +
  geom_col(fill = "#3B7EA1", width = 0.72) +
  geom_text(aes(label = selected_outer_folds), hjust = -0.15, size = 3.2) +
  scale_x_continuous(
    limits = c(0, outer_v * outer_repeats + 2),
    breaks = seq(0, outer_v * outer_repeats, 5),
    expand = c(0, 0)
  ) +
  labs(
    title = "Algorithm selected within outer training sets",
    subtitle = "Selection used inner results only",
    x = "Number of outer folds selected", y = NULL
  ) +
  theme_classic(base_size = 11, base_family = "sans") +
  theme(plot.title = element_text(face = "bold"))
save_plot(p_choice, "Figure_2_algorithm_selection_frequency", 8.5, 5.5)

# 14. Save complete result object -----------------------------
results <- list(
  settings = configuration,
  selection_rule = selection_rule,
  tuning_plan = tuning_plan,
  audit = audit,
  candidate_predictors = candidate_predictors,
  prespecified_exclusions = correlation_drop,
  outer_folds = raw_results$outer_folds,
  inner_lasso_summary = raw_results$inner_lasso_summary,
  inner_lasso_variables = raw_results$inner_lasso_variables,
  outer_lasso_summary = raw_results$outer_lasso_summary,
  outer_lasso_variables = raw_results$outer_lasso_variables,
  lasso_selection_frequency = lasso_selection_frequency,
  inner_model_summary = raw_results$inner_model_summary,
  inner_best_parameters = raw_results$inner_best_parameters,
  tuning_candidate_grids = raw_results$tuning_grids,
  tuning_metrics = raw_results$tuning_metrics,
  selected_model_by_outer_fold = raw_results$outer_choices |>
    filter(selected),
  algorithm_selection_frequency = outer_choice_frequency,
  all_model_outer_predictions = raw_results$outer_predictions_all,
  selected_pipeline_outer_predictions = raw_results$outer_predictions_selected,
  all_model_fold_metrics = all_model_fold_metrics,
  all_model_patient_oof = all_model_patient_oof,
  all_model_summary = all_model_summary,
  model_ranking = model_ranking,
  selected_model = selected_model,
  selected_pipeline_fold_metrics = selected_fold_metrics,
  selected_pipeline_patient_oof = selected_patient_oof,
  selected_pipeline_performance = selected_pipeline_performance,
  outer_parameters_used = raw_results$outer_parameters_used,
  hyperparameter_boundary_details = boundary_details,
  hyperparameter_boundary_summary = boundary_summary,
  outer_assessment_membership = raw_results$membership,
  errors = raw_results$errors,
  tuning_notes = raw_results$tuning_notes,
  session_info = sessionInfo()
)

results_path <- file.path(out_dir, "fully_nested_lasso_9models_results.rds")
saveRDS(results, results_path, compress = "xz", version = 3)
if (!file.exists(results_path)) stop("Failed to save the analysis results.")

message(
  "Fully nested analysis completed successfully.\n",
  "Automatically selected final algorithm: ", selected_model, "\n",
  "Selected-pipeline pooled OOF AUC: ",
  signif(selected_pipeline_performance$pooled_oof_auc, 4), "\n",
  "Results saved to: ",
  normalizePath(results_path, winslash = "/", mustWork = TRUE)
)
