# ============================================================
# Supplementary Table S6
# Case-level external-validation error analysis of the locked RF model
#
# Outputs:
#   1. Table_S6_external_error_cases.csv
#   2. Table_S6_external_error_cases.docx
#   3. Table_S6_error_counts.csv
#   4. Table_S6_caption_and_footnote.txt
#
# The original project paths are intentionally preserved.
# The external threshold is fixed at 0.50 and is not optimized on the
# external-validation cohort.
# ============================================================

# install.packages(c("dplyr", "officer", "flextable", "workflows"))

required_packages <- c("dplyr", "tibble", "officer", "flextable", "workflows")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install required packages: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(officer)
  library(flextable)
})

# -----------------------------
# 1. Paths and fixed settings
# -----------------------------

root <- "C:/Users/Lenovo/Desktop/impacted_pr"
bundle_path <- file.path(
  root, "output", "final_selected_model",
  "final_random_forest_model_bundle.rds"
)
external_path <- file.path(root, "data", "data_wb.csv")
output_dir <- file.path(root, "output", "Table_S6_external_error_cases")

fixed_threshold <- 0.50
expected_external_n <- 101L
expected_fp <- 8L
expected_fn <- 3L

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

missing_files <- c(bundle_path, external_path)
missing_files <- missing_files[!file.exists(missing_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing input file(s):\n",
    paste(missing_files, collapse = "\n"),
    call. = FALSE
  )
}

# -----------------------------
# 2. Read and validate locked RF
# -----------------------------

bundle <- readRDS(bundle_path)
required_bundle_fields <- c(
  "fitted_workflow", "selected_model", "selected_predictors"
)

if (!all(required_bundle_fields %in% names(bundle)) ||
    !identical(as.character(bundle$selected_model), "random_forest") ||
    !inherits(bundle$fitted_workflow, "workflow")) {
  stop(
    "The supplied model bundle is not the locked random-forest workflow.",
    call. = FALSE
  )
}

predictors <- as.character(bundle$selected_predictors)
expected_predictors <- c(
  "flank_pain",
  "stone_attenuation_hu",
  "proximal_ureter_width",
  "ureteral_wall_thickness",
  "stone_length",
  "hydronephrosis_area"
)

if (length(predictors) != 6L || anyDuplicated(predictors) ||
    !setequal(predictors, expected_predictors)) {
  stop(
    "The locked RF does not contain the expected six final predictors.",
    call. = FALSE
  )
}

# Check that any threshold stored in the bundle agrees with the prespecified
# external threshold. A threshold optimized on the external cohort is not used.
configuration <- bundle$configuration
threshold_keys <- c(
  "classification_threshold", "decision_threshold",
  "probability_threshold", "fixed_threshold"
)
stored_thresholds <- numeric(0)

if (is.data.frame(configuration) &&
    all(c("setting", "value") %in% names(configuration))) {
  stored_thresholds <- c(
    stored_thresholds,
    suppressWarnings(as.numeric(as.character(
      configuration$value[configuration$setting %in% threshold_keys]
    )))
  )
}

for (field in threshold_keys) {
  if (field %in% names(bundle)) {
    stored_thresholds <- c(
      stored_thresholds,
      suppressWarnings(as.numeric(bundle[[field]]))
    )
  }
}

stored_thresholds <- unique(stored_thresholds[is.finite(stored_thresholds)])
if (length(stored_thresholds) > 1L ||
    (length(stored_thresholds) == 1L &&
     abs(stored_thresholds - fixed_threshold) > 1e-12)) {
  stop(
    "The locked bundle contains a threshold different from the fixed 0.50 ",
    "external-reporting threshold.",
    call. = FALSE
  )
}

# -----------------------------
# 3. Prepare external-validation cohort
# -----------------------------

raw_names <- c(
  "Flank Pain" = "flank_pain",
  "Stone HU" = "stone_attenuation_hu",
  "WUPC" = "proximal_ureter_width",
  "UWT" = "ureteral_wall_thickness",
  "Stone length" = "stone_length",
  "APH" = "hydronephrosis_area",
  "Impacted stone" = "impacted_stone"
)

prepare_external <- function(data) {
  data <- as.data.frame(data, check.names = FALSE)

  for (from in names(raw_names)) {
    to <- raw_names[[from]]
    if (from %in% names(data) && !to %in% names(data)) {
      names(data)[names(data) == from] <- to
    }
  }

  required_columns <- c(predictors, "impacted_stone")
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      "External data are missing: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  if (!nrow(data)) {
    stop("The external-validation data contain no rows.", call. = FALSE)
  }

  for (variable in setdiff(predictors, "flank_pain")) {
    data[[variable]] <- suppressWarnings(
      as.numeric(as.character(data[[variable]]))
    )
    if (any(!is.finite(data[[variable]]))) {
      stop(
        "Invalid values found in external predictor: ", variable,
        call. = FALSE
      )
    }
  }

  flank_pain <- trimws(as.character(data$flank_pain))
  if (!all(flank_pain %in% c("No", "Yes"))) {
    stop(
      "flank_pain must contain only 'No' and 'Yes'.",
      call. = FALSE
    )
  }
  data$flank_pain <- factor(flank_pain, levels = c("No", "Yes"))

  outcome <- trimws(as.character(data$impacted_stone))
  outcome[outcome == "Yes"] <- "Impacted"
  outcome[outcome == "No"] <- "Non-impacted"
  if (!all(outcome %in% c("Impacted", "Non-impacted"))) {
    stop(
      "impacted_stone must be coded as Impacted/Non-impacted or Yes/No.",
      call. = FALSE
    )
  }
  data$impacted_stone <- factor(
    outcome,
    levels = c("Impacted", "Non-impacted")
  )

  data
}

external <- prepare_external(
  read.csv(
    external_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
)

if (nrow(external) != expected_external_n) {
  stop(
    "Expected ", expected_external_n,
    " external-validation patients but found ", nrow(external), ".",
    call. = FALSE
  )
}

# -----------------------------
# 4. Locked RF predictions
# -----------------------------

predict_risk <- function(object, newdata) {
  prediction <- predict(object, new_data = newdata, type = "prob")
  if (!".pred_Impacted" %in% names(prediction)) {
    stop(
      "The locked RF did not return .pred_Impacted.",
      call. = FALSE
    )
  }
  as.numeric(prediction$.pred_Impacted)
}

probability <- predict_risk(
  bundle$fitted_workflow,
  external[, predictors, drop = FALSE]
)

if (length(probability) != nrow(external) ||
    any(!is.finite(probability)) ||
    any(probability < 0 | probability > 1)) {
  stop(
    "External RF probabilities must be finite and within [0, 1].",
    call. = FALSE
  )
}

true_outcome <- as.character(external$impacted_stone)
predicted_outcome <- ifelse(
  probability >= fixed_threshold,
  "Impacted",
  "Non-impacted"
)

error_type <- dplyr::case_when(
  true_outcome == "Non-impacted" & predicted_outcome == "Impacted" ~ "FP",
  true_outcome == "Impacted" & predicted_outcome == "Non-impacted" ~ "FN",
  TRUE ~ NA_character_
)

error_cases <- external %>%
  mutate(
    true_outcome = true_outcome,
    predicted_outcome = predicted_outcome,
    predicted_probability = probability,
    error_type = error_type
  ) %>%
  filter(!is.na(error_type)) %>%
  arrange(
    factor(error_type, levels = c("FP", "FN")),
    desc(predicted_probability)
  ) %>%
  group_by(error_type) %>%
  mutate(case_id = sprintf("E-%s%02d", error_type, row_number())) %>%
  ungroup()

error_counts <- error_cases %>%
  count(error_type, name = "n") %>%
  right_join(
    tibble::tibble(error_type = c("FP", "FN")),
    by = "error_type"
  ) %>%
  mutate(n = ifelse(is.na(n), 0L, n)) %>%
  arrange(factor(error_type, levels = c("FP", "FN")))

if (nrow(error_cases) != expected_fp + expected_fn ||
    error_counts$n[error_counts$error_type == "FP"] != expected_fp ||
    error_counts$n[error_counts$error_type == "FN"] != expected_fn) {
  stop(
    "Expected exactly 8 FP and 3 FN cases, but the locked RF produced: ",
    paste(error_counts$error_type, error_counts$n, collapse = "; "),
    call. = FALSE
  )
}

# -----------------------------
# 5. Machine-readable Table S6
# -----------------------------

csv_table <- error_cases %>%
  transmute(
    case_id,
    error_type,
    true_outcome,
    predicted_outcome,
    predicted_probability,
    classification_threshold = fixed_threshold,
    flank_pain = as.character(flank_pain),
    stone_attenuation_hu,
    proximal_ureter_width,
    ureteral_wall_thickness,
    stone_length,
    hydronephrosis_area
  )

write.csv(
  csv_table,
  file = file.path(output_dir, "Table_S6_external_error_cases.csv"),
  row.names = FALSE,
  na = ""
)

write.csv(
  error_counts,
  file = file.path(output_dir, "Table_S6_error_counts.csv"),
  row.names = FALSE,
  na = ""
)

# -----------------------------
# 6. Publication-grade Word table
# -----------------------------

format_number <- function(x, digits = 2L) {
  ifelse(
    is.finite(x),
    formatC(x, format = "f", digits = digits),
    "NR"
  )
}

word_table <- error_cases %>%
  transmute(
    `Case ID` = case_id,
    `Error type` = error_type,
    `True outcome` = true_outcome,
    `Predicted outcome` = predicted_outcome,
    `P(Impacted)` = format_number(predicted_probability, 3L),
    `Flank pain` = as.character(flank_pain),
    `Stone attenuation\n(HU)` = format_number(stone_attenuation_hu, 0L),
    `Proximal ureter width\n(mm)` = format_number(proximal_ureter_width, 2L),
    `UWT\n(mm)` = format_number(ureteral_wall_thickness, 2L),
    `Stone length\n(mm)` = format_number(stone_length, 2L),
    `APH\n(mm²)` = format_number(hydronephrosis_area, 2L)
  )

ft <- flextable::flextable(word_table)
ft <- flextable::theme_booktabs(ft)
ft <- flextable::set_table_properties(ft, layout = "fixed", width = 1)
ft <- flextable::font(ft, fontname = "Arial", part = "all")
ft <- flextable::fontsize(ft, size = 7.5, part = "all")
ft <- flextable::bold(ft, part = "header")
ft <- flextable::color(ft, color = "white", part = "header")
ft <- flextable::bg(ft, bg = "steelblue4", part = "header")
ft <- flextable::align(ft, align = "center", part = "all")
ft <- flextable::align(ft, j = 1, align = "left", part = "all")
ft <- flextable::padding(ft, padding = 2, part = "all")
ft <- flextable::valign(ft, valign = "center", part = "all")

column_widths <- c(
  0.72, 0.68, 1.00, 1.10, 0.82, 0.78,
  0.95, 1.05, 0.78, 0.85, 0.82
)
ft <- flextable::width(
  ft,
  j = seq_along(column_widths),
  width = column_widths
)

fp_rows <- which(word_table$`Error type` == "FP")
fn_rows <- which(word_table$`Error type` == "FN")
if (length(fp_rows)) {
  ft <- flextable::bg(ft, i = fp_rows, bg = "lemonchiffon", part = "body")
}
if (length(fn_rows)) {
  ft <- flextable::bg(ft, i = fn_rows, bg = "mistyrose", part = "body")
}
ft <- flextable::hline(
  ft,
  border = officer::fp_border(color = "lightsteelblue2", width = 0.4),
  part = "body"
)
ft <- flextable::hline_bottom(
  ft,
  border = officer::fp_border(color = "steelblue4", width = 0.8),
  part = "all"
)

caption_text <- paste(
  "Table S6. Case-level external-validation error analysis of the locked",
  "random-forest model. The table includes all false-positive (FP) and",
  "false-negative (FN) cases observed in the external-validation cohort",
  "(n = 101). The positive event was impacted ureteral stone.",
  sep = " "
)

note_text <- paste(
  "Predicted probabilities represent P(Impacted). The classification",
  "threshold was fixed at 0.50 for external reporting and was not optimized",
  "using the external cohort. FP denotes a non-impacted stone classified as",
  "impacted; FN denotes an impacted stone classified as non-impacted.",
  "Case IDs were newly generated for de-identification. Continuous variables",
  "are reported in their original clinical units. Flank pain is coded as",
  "No/Yes. The six predictors correspond to the locked RF model inputs.",
  sep = " "
)

small_text <- officer::fp_text(
  font.family = "Arial", font.size = 8, color = "grey20"
)
title_text <- officer::fp_text(
  font.family = "Arial", font.size = 12, bold = TRUE, color = "black"
)

word_file <- file.path(
  output_dir,
  "Table_S6_external_error_cases.docx"
)

doc <- officer::read_docx()
doc <- officer::body_add_fpar(
  doc,
  officer::fpar(
    officer::ftext(
      "Table S6. External-validation error cases",
      prop = title_text
    )
  )
)
doc <- flextable::body_add_flextable(doc, ft)
doc <- officer::body_add_par(doc, "")
doc <- officer::body_add_fpar(
  doc,
  officer::fpar(officer::ftext(caption_text, prop = small_text))
)
doc <- officer::body_add_fpar(
  doc,
  officer::fpar(officer::ftext(note_text, prop = small_text))
)

doc <- officer::body_end_block_section(
  doc,
  officer::block_section(
    officer::prop_section(
      page_size = officer::page_size(orient = "landscape"),
      page_margins = officer::page_mar(
        top = 0.45, bottom = 0.45, left = 0.45, right = 0.45,
        header = 0.2, footer = 0.2
      )
    )
  )
)

print(doc, target = word_file)

writeLines(
  c(caption_text, "", note_text),
  con = file.path(output_dir, "Table_S6_caption_and_footnote.txt"),
  useBytes = TRUE
)

message("Table S6 Word file saved to: ", normalizePath(word_file, winslash = "/"))
message("Table S6 outputs saved to: ", normalizePath(output_dir, winslash = "/"))
