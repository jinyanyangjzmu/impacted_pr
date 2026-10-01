# ============================================================
# Impacted stone study
# Finalize one prespecified model in the complete development cohort
#
# Workflow:
#   1. Refit LASSO on the complete development cohort.
#   2. Tune only the prespecified model by repeated cross-validation.
#   3. Refit the selected workflow on all development observations.
#   4. Save the locked workflow and a reproducibility bundle.
#
# The external cohort is never read in this script.
# Final-tuning metrics are development summaries, not validation estimates.
#
# Usage:
# Rscript impacted_finalize_selected_model.R \
#   [development_data.rds] [nested_results.rds] [output_dir] [model]
# ============================================================

# 1. User settings -------------------------------------------
script_version <- "1.0.0"
seed <- 2026L
RNGkind("L'Ecuyer-CMRG")

outcome <- "impacted_stone"
id_var <- "patient_id"
positive <- "Impacted"
negative <- "Non-impacted"

final_model <- "xgboost"  # Change only after manual model selection.
tuning_v <- 5L
tuning_repeats <- 5L
grid_design <- "max_min_l2"

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

correlation_drop <- c(
  "stone_width",
  "hu_above_below_ratio",
  "alcohol_use",
  "hydronephrosis_grade",
  "ureteral_wall_area"
)

# 2. Paths and packages --------------------------------------
args <- commandArgs(trailingOnly = TRUE)

default_project_dir <- "C:/Users/Lenovo/Desktop/impacted_pr"
data_path <- if (length(args) >= 1L) args[[1]] else file.path(
  default_project_dir, "data", "impacted_clean.rds"
)
nested_path <- if (length(args) >= 2L) args[[2]] else file.path(
  default_project_dir, "output", "nested_lasso_9models_publication",
  "nested_lasso_9models_results.rds"
)
out_dir <- if (length(args) >= 3L) args[[3]] else file.path(
  default_project_dir, "output", "final_selected_model"
)
if (length(args) >= 4L) final_model <- args[[4]]

if (!final_model %in% names(grid_sizes)) {
  stop("Unsupported final_model: ", final_model)
}
if (!file.exists(data_path)) stop("Development data not found: ", data_path)
if (!file.exists(nested_path)) stop("Nested-CV results not found: ", nested_path)

engine_packages <- list(
  logistic = character(),
  elastic_net = "glmnet",
  decision_tree = "rpart",
  random_forest = "ranger",
  svm_rbf = "kernlab",
  knn = "kknn",
  lightgbm = c("bonsai", "lightgbm"),
  mlp = "nnet",
  xgboost = "xgboost"
)
required_pkgs <- unique(c(
  "tidymodels", "glmnet", "doParallel", "readr",
  engine_packages[[final_model]]
))
missing_pkgs <- required_pkgs[
  !vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_pkgs)) {
  stop("Install required packages first: ", paste(missing_pkgs, collapse = ", "))
}

suppressPackageStartupMessages({
  library(tidymodels)
  library(glmnet)
  library(doParallel)
  library(readr)
})
if (final_model == "lightgbm") {
  suppressPackageStartupMessages(library(bonsai))
}
tidymodels_prefer()

data_path <- normalizePath(data_path, winslash = "/", mustWork = TRUE)
nested_path <- normalizePath(nested_path, winslash = "/", mustWork = TRUE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, winslash = "/", mustWork = TRUE)

# 3. Data and provenance checks ------------------------------
dat <- readRDS(data_path) |> as.data.frame(check.names = FALSE)
nested_results <- readRDS(nested_path)

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

data_md5 <- unname(tools::md5sum(data_path))
nested_md5 <- unname(tools::md5sum(nested_path))

if (!is.null(nested_results$candidate_predictors) &&
    !identical(candidate_predictors, nested_results$candidate_predictors)) {
  stop("Candidate predictors differ from the nested-CV analysis.")
}
if (!is.null(nested_results$prespecified_exclusions) &&
    !identical(correlation_drop, nested_results$prespecified_exclusions)) {
  stop("Prespecified exclusions differ from the nested-CV analysis.")
}

recorded_md5 <- nested_results$settings |>
  filter(setting == "data_md5") |>
  pull(value)
if (length(recorded_md5) == 1L && !identical(data_md5, recorded_md5)) {
  stop("Development-data MD5 differs from the nested-CV analysis.")
}

selection_record <- nested_results$model_ranking |>
  filter(model == final_model)
if (nrow(selection_record) != 1L) {
  stop("The selected model is absent or duplicated in model_ranking.")
}
if (selection_record$failed_outer_folds[[1]] != 0L ||
    !isTRUE(selection_record$complete_oof_coverage[[1]])) {
  stop("The selected model did not complete leakage-free outer validation.")
}

# 4. LASSO selection -----------------------------------------
make_foldid <- function(y, v, seed_value) {
  set.seed(seed_value)
  fold_id <- integer(length(y))
  for (level in sort(unique(y))) {
    index <- which(y == level)
    fold_id[index] <- sample(rep(seq_len(v), length.out = length(index)))
  }
  fold_id
}

select_lasso_variables <- function(data, predictors, seed_value) {
  formula <- reformulate(predictors, response = outcome)
  terms_obj <- terms(formula, data = data)
  matrix_all <- model.matrix(terms_obj, data)
  assignment <- attr(matrix_all, "assign")
  term_labels <- attr(terms_obj, "term.labels")

  keep <- colnames(matrix_all) != "(Intercept)"
  x <- matrix_all[, keep, drop = FALSE]
  dummy_map <- tibble(
    term = colnames(x),
    predictor = term_labels[assignment[keep]]
  )

  nonzero_variance <- apply(x, 2, sd) > 0
  x <- x[, nonzero_variance, drop = FALSE]
  dummy_map <- dummy_map[nonzero_variance, , drop = FALSE]
  if (!ncol(x)) stop("No non-zero-variance predictors remained for LASSO.")

  y <- as.integer(data[[outcome]] == positive)
  fold_id <- make_foldid(y, tuning_v, seed_value)

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

  extract_terms <- function(lambda) {
    coefficients <- as.matrix(coef(cv_fit, s = lambda))
    tibble(
      term = rownames(coefficients),
      coefficient = coefficients[, 1]
    ) |>
      filter(term != "(Intercept)", coefficient != 0) |>
      left_join(dummy_map, by = "term")
  }

  lambda_used <- cv_fit$lambda.1se
  lambda_rule <- "lambda.1se"
  selected_terms <- extract_terms(lambda_used)

  if (!nrow(selected_terms)) {
    lambda_used <- cv_fit$lambda.min
    lambda_rule <- "lambda.min_fallback"
    selected_terms <- extract_terms(lambda_used)
  }
  if (!nrow(selected_terms)) stop("LASSO selected no variables at lambda.min.")

  selected_predictors <- intersect(
    predictors, unique(selected_terms$predictor)
  )

  list(
    selected = selected_predictors,
    selected_terms = selected_terms,
    lambda_rule = lambda_rule,
    lambda_used = lambda_used,
    lambda_1se = cv_fit$lambda.1se,
    lambda_min = cv_fit$lambda.min,
    cv_fit = cv_fit
  )
}

lasso <- select_lasso_variables(
  dat,
  candidate_predictors,
  seed_value = seed + 10000L
)
selected <- lasso$selected

lasso_summary <- tibble(
  lambda_rule = lasso$lambda_rule,
  lambda_used = lasso$lambda_used,
  lambda_1se = lasso$lambda_1se,
  lambda_min = lasso$lambda_min,
  n_selected_predictors = length(selected),
  n_selected_terms = nrow(lasso$selected_terms)
)

selection_frequency <- nested_results$lasso_selection_frequency |>
  select(predictor, selected_outer_folds, successful_lasso_folds,
         selection_frequency)

selected_predictor_table <- tibble(
  selection_order = seq_along(selected),
  predictor = selected
) |>
  left_join(selection_frequency, by = "predictor")

# 5. Recipe and selected-model definition --------------------
count_design_columns <- function(data, predictors) {
  x <- model.matrix(reformulate(predictors), data)
  x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  if (ncol(x)) x <- x[, apply(x, 2, sd) > 0, drop = FALSE]
  max(1L, ncol(x))
}

make_recipe <- function(data, predictors, normalize = FALSE) {
  rec <- recipe(reformulate(predictors, response = outcome), data = data) |>
    step_unknown(all_nominal_predictors(), new_level = "unknown") |>
    step_novel(all_nominal_predictors(), new_level = "novel") |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors())

  if (normalize) rec <- rec |> step_normalize(all_numeric_predictors())
  rec
}

make_model <- function(model_name, p) {
  mtry_value <- if (p > 1L) tune() else 1L
  mtry_parameter <- if (p > 1L) list(dials::mtry(c(1L, p))) else list()
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

  switch(
    model_name,
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
        mtry = !!mtry_value, min_n = tune(), trees = 1000L
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

# 6. Final tuning and refit -----------------------------------
model_data <- dat[, c(id_var, outcome, selected), drop = FALSE]
p_design <- count_design_columns(model_data, selected)
definition <- make_model(final_model, p_design)
model_recipe <- make_recipe(
  model_data,
  selected,
  normalize = final_model %in% scale_sensitive_models
)
base_workflow <- workflow() |>
  add_recipe(model_recipe) |>
  add_model(definition$spec)

tune_result <- NULL
tuning_grid <- tibble()
tuning_metrics <- tibble()
tuning_notes <- tibble()

if (is.null(definition$params)) {
  best_parameters <- tibble()
  final_workflow <- base_workflow
} else {
  if (any(table(model_data[[outcome]]) < tuning_v)) {
    stop("An outcome class has fewer observations than tuning_v.")
  }

  set.seed(seed + 20000L)
  tuning_folds <- vfold_cv(
    model_data,
    v = tuning_v,
    repeats = tuning_repeats,
    strata = impacted_stone
  )

  set.seed(seed + 30000L)
  tuning_grid <- dials::grid_space_filling(
    definition$params,
    size = unname(grid_sizes[[final_model]]),
    type = grid_design
  ) |>
    mutate(candidate_id = row_number(), .before = 1)

  available_cores <- parallel::detectCores(logical = FALSE)
  if (is.na(available_cores)) available_cores <- 2L
  n_cores <- max(1L, min(8L, available_cores - 1L))
  cluster <- parallel::makePSOCKcluster(n_cores)
  doParallel::registerDoParallel(cluster)

  tune_result <- tryCatch(
    {
      set.seed(seed + 30000L)
      tune_grid(
        base_workflow,
        resamples = tuning_folds,
        grid = tuning_grid |> select(-candidate_id),
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
    },
    finally = {
      parallel::stopCluster(cluster)
      foreach::registerDoSEQ()
    }
  )

  tuning_metrics <- collect_metrics(tune_result)
  tuning_notes <- collect_notes(tune_result)
  if (!any(
    tuning_metrics$.metric == "roc_auc" & is.finite(tuning_metrics$mean)
  )) {
    stop("No valid final-tuning ROC AUC estimates were produced.")
  }

  best_parameters <- select_best(tune_result, metric = "roc_auc")
  final_workflow <- finalize_workflow(base_workflow, best_parameters)
}

set.seed(seed + 40000L)
fitted_workflow <- fit(final_workflow, data = model_data)

# Apparent training metrics are saved only as a fitting check.
probability_column <- paste0(".pred_", make.names(positive))
training_probability <- predict(
  fitted_workflow,
  new_data = model_data,
  type = "prob"
)
if (!probability_column %in% names(training_probability)) {
  stop("Positive-class probability column was not found.")
}
probability <- training_probability[[probability_column]]
if (any(!is.finite(probability)) || any(probability < 0 | probability > 1)) {
  stop("The final model returned invalid probabilities.")
}

truth01 <- as.integer(model_data[[outcome]] == positive)
bounded_probability <- pmin(pmax(probability, 1e-6), 1 - 1e-6)
linear_predictor <- qlogis(bounded_probability)

training_apparent_metrics <- tibble(
  dataset = "Complete development cohort (apparent only)",
  n = nrow(model_data),
  roc_auc = roc_auc_vec(
    model_data[[outcome]], probability, event_level = "first"
  ),
  brier = mean((truth01 - probability)^2),
  calibration_intercept = unname(coef(
    glm(truth01 ~ offset(linear_predictor), family = binomial())
  )[1]),
  calibration_slope = unname(coef(
    glm(truth01 ~ linear_predictor, family = binomial())
  )[2])
)

# 7. Outputs --------------------------------------------------
configuration <- tibble(
  setting = c(
    "script_version", "selected_model", "seed", "rng_kind",
    "data_file", "data_md5", "nested_results_file", "nested_results_md5",
    "positive_class", "negative_class", "lasso_alpha",
    "lasso_primary_rule", "lasso_empty_fallback", "lasso_cv_folds",
    "tuning_v", "tuning_repeats", "tuning_selection_metric",
    "grid_design", "grid_size", "normalized_predictors",
    "external_data_accessed"
  ),
  value = as.character(c(
    script_version, final_model, seed, paste(RNGkind(), collapse = "; "),
    data_path, data_md5, nested_path, nested_md5,
    positive, negative, 1,
    "lambda.1se", "lambda.min", tuning_v,
    tuning_v, tuning_repeats, "Mean repeated-CV ROC AUC",
    grid_design, unname(grid_sizes[[final_model]]),
    final_model %in% scale_sensitive_models,
    "No"
  ))
)

write_table <- function(data, filename) {
  if (!ncol(data) || !nrow(data)) data <- tibble(message = "No records")
  write_excel_csv(data, file.path(out_dir, filename))
}

write_table(lasso_summary, "21_final_lasso_summary.csv")
write_table(
  selected_predictor_table,
  "22_final_lasso_selected_variables.csv"
)
write_table(lasso$selected_terms, "23_final_lasso_selected_terms.csv")
write_table(tuning_grid, "24_final_tuning_candidate_grid.csv")
write_table(tuning_metrics, "25_final_tuning_metrics.csv")
write_table(best_parameters, "26_final_best_parameters.csv")
write_table(
  training_apparent_metrics,
  "27_final_training_apparent_metrics.csv"
)
write_table(configuration, "28_finalization_configuration.csv")
write_table(tuning_notes, "29_final_tuning_notes.csv")
writeLines(
  capture.output(sessionInfo()),
  file.path(out_dir, "30_sessionInfo.txt")
)

workflow_path <- file.path(
  out_dir,
  paste0("final_", final_model, "_workflow.rds")
)
bundle_path <- file.path(
  out_dir,
  paste0("final_", final_model, "_model_bundle.rds")
)

saveRDS(fitted_workflow, workflow_path, compress = "xz", version = 3)

model_bundle <- list(
  fitted_workflow = fitted_workflow,
  selected_model = final_model,
  selected_predictors = selected,
  selected_predictor_table = selected_predictor_table,
  lasso_summary = lasso_summary,
  lasso_selected_terms = lasso$selected_terms,
  lasso_cv_fit = lasso$cv_fit,
  best_parameters = best_parameters,
  tuning_grid = tuning_grid,
  tuning_metrics = tuning_metrics,
  tuning_notes = tuning_notes,
  tuning_result = tune_result,
  nested_cv_selection_record = selection_record,
  training_apparent_metrics = training_apparent_metrics,
  outcome = list(
    name = outcome,
    positive = positive,
    negative = negative,
    factor_levels = levels(dat[[outcome]])
  ),
  configuration = configuration,
  session_info = sessionInfo()
)
saveRDS(model_bundle, bundle_path, compress = "xz", version = 3)

# Reload the locked workflow and verify that it still predicts probabilities.
locked_workflow <- readRDS(workflow_path)
verification <- predict(
  locked_workflow,
  new_data = model_data[seq_len(min(5L, nrow(model_data))), , drop = FALSE],
  type = "prob"
)
if (!probability_column %in% names(verification) ||
    any(!is.finite(verification[[probability_column]]))) {
  stop("Saved-workflow verification failed.")
}

message(
  "Finalization completed successfully.\n",
  "Selected model: ", final_model, "\n",
  "Selected predictors (", length(selected), "): ",
  paste(selected, collapse = ", "), "\n",
  "Locked workflow: ", workflow_path, "\n",
  "Reproducibility bundle: ", bundle_path
)
