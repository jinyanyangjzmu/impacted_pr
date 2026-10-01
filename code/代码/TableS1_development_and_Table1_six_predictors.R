# Tables for the impacted ureteral stone manuscript
# Run on the investigator's Windows computer:
#   source("C:/Users/Lenovo/Desktop/impacted_pr/TableS1_development_and_Table1_six_predictors.R")
# Install once if necessary: install.packages(c("flextable", "officer"))
# Outputs: two editable Word tables and two matching CSV files.

# ---- Paths and editorial option ------------------------------------------
project_dir <- "C:/Users/Lenovo/Desktop/impacted_pr"
development_file <- file.path(project_dir, "data", "impacted_clean.rds")
external_file <- file.path(project_dir, "data", "data_wb.csv")
model_file <- file.path(project_dir, "output", "final_selected_model",
                        "final_random_forest_model_bundle.rds")
output_dir <- file.path(project_dir, "output", "nested_lasso_9models",
                        "manuscript_tables")

# TRUE reproduces the full-variable Total column of the old manuscript's Table 1.
# Change to FALSE if the journal wants the six Table 1 predictors omitted from S1.
include_model_variables_in_s1 <- TRUE

needed_packages <- c("flextable", "officer")
missing_packages <- needed_packages[!vapply(needed_packages,
                                            requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install R packages: install.packages(c(",
       paste(sprintf('"%s"', missing_packages), collapse = ", "), "))")
}
inputs <- c(development_file, external_file, model_file)
if (any(!file.exists(inputs))) {
  stop("Input files missing:\n", paste(inputs[!file.exists(inputs)], collapse = "\n"))
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Data and locked-predictor verification -------------------------------
development <- as.data.frame(readRDS(development_file), check.names = FALSE)
external <- read.csv(external_file, check.names = FALSE, stringsAsFactors = FALSE,
                     fileEncoding = "UTF-8-BOM")
bundle <- readRDS(model_file)

locked <- c("flank_pain", "stone_attenuation_hu", "proximal_ureter_width",
            "ureteral_wall_thickness", "stone_length", "hydronephrosis_area")
if (!identical(as.character(bundle$selected_model), "random_forest") ||
    !setequal(as.character(bundle$selected_predictors), locked)) {
  stop("The selected model or six predictors differ from the locked RF bundle.")
}
provenance <- bundle$configuration
if (!is.data.frame(provenance) ||
    !all(c("setting", "value") %in% names(provenance))) {
  stop("The RF bundle does not contain a verifiable configuration record.")
}
locked_md5 <- as.character(provenance$value[provenance$setting == "data_md5"])
if (length(locked_md5) != 1L ||
    unname(tools::md5sum(development_file)) != locked_md5) {
  stop("Development RDS differs from the data used to lock the final RF.")
}

external_names <- c("Name" = "patient_id", "Flank Pain" = "flank_pain",
                    "Stone HU" = "stone_attenuation_hu",
                    "WUPC" = "proximal_ureter_width",
                    "UWT" = "ureteral_wall_thickness",
                    "Stone length" = "stone_length",
                    "APH" = "hydronephrosis_area",
                    "Impacted stone" = "impacted_stone")
for (old in names(external_names)) {
  if (old %in% names(external)) names(external)[names(external) == old] <- external_names[[old]]
}

required <- c(locked, "impacted_stone")
for (nm in c("development", "external")) {
  d <- get(nm)
  missing <- setdiff(required, names(d))
  if (length(missing)) stop(nm, " missing columns: ", paste(missing, collapse = ", "))
}
if (nrow(development) != 545L || nrow(external) != 101L) {
  stop("Unexpected cohort sizes: development = ", nrow(development),
       "; external = ", nrow(external), ". Check the input files.")
}

clean_chr <- function(x) {
  y <- trimws(as.character(x))
  y[y %in% c("", "NA", "N/A", "NULL")] <- NA_character_
  y
}
numeric_value <- function(x, variable) {
  y <- suppressWarnings(as.numeric(clean_chr(x)))
  if (any(!is.na(clean_chr(x)) & is.na(y)) || any(!is.na(y) & !is.finite(y))) {
    stop("Non-numeric or non-finite data in ", variable)
  }
  y
}
category_value <- function(x, allowed, variable) {
  y <- clean_chr(x)
  unknown <- setdiff(unique(stats::na.omit(y)), allowed)
  if (length(unknown)) {
    stop("Unexpected category in ", variable, ": ", paste(unknown, collapse = ", "))
  }
  y
}
for (nm in c("development", "external")) {
  d <- get(nm)
  for (v in setdiff(locked, "flank_pain")) d[[v]] <- numeric_value(d[[v]], v)
  d$flank_pain <- category_value(d$flank_pain, c("No", "Yes"), "flank_pain")
  d$impacted_stone <- category_value(d$impacted_stone,
                                     c("Non-impacted", "Impacted", "No", "Yes"),
                                     "impacted_stone")
  d$impacted_stone[d$impacted_stone == "Yes"] <- "Impacted"
  d$impacted_stone[d$impacted_stone == "No"] <- "Non-impacted"
  assign(nm, d)
}
if (sum(development$impacted_stone == "Impacted", na.rm = TRUE) != 209L ||
    sum(external$impacted_stone == "Impacted", na.rm = TRUE) != 38L) {
  stop("Outcome counts differ from the previous manuscript (209 / 38). Reconcile datasets first.")
}

# ---- Formatters: valid denominators and explicit missingness -------------
fmt_n <- function(x, level, cohort_n) {
  observed <- !is.na(x)
  n <- sum(observed & x == level)
  denominator <- sum(observed)
  if (!denominator) return("NA")
  value <- sprintf("%d (%.1f%%)", n, 100 * n / denominator)
  if (denominator != cohort_n) value <- paste0(value, " [n=", denominator, "]")
  value
}
fmt_median <- function(x, digits = 2L, cohort_n) {
  x <- x[!is.na(x)]
  if (!length(x)) return("NA")
  q <- stats::quantile(x, c(.25, .50, .75), names = FALSE, type = 7)
  value <- sprintf(paste0("%.", digits, "f [%.", digits,
                          "f, %.", digits, "f]"), q[2L], q[1L], q[3L])
  if (length(x) != cohort_n) value <- paste0(value, " [n=", length(x), "]")
  value
}
standardized_difference <- function(x, y, positive = NULL) {
  if (!is.null(positive)) {
    a <- mean(x[!is.na(x)] == positive)
    b <- mean(y[!is.na(y)] == positive)
    denom <- sqrt((a * (1 - a) + b * (1 - b)) / 2)
  } else {
    x <- x[!is.na(x)]
    y <- y[!is.na(y)]
    a <- mean(x)
    b <- mean(y)
    denom <- sqrt((stats::var(x) + stats::var(y)) / 2)
  }
  if (!is.finite(denom) || denom == 0) return("NA")
  sprintf("%.2f", abs(a - b) / denom)
}

# ---- Supplementary Table S1: old manuscript's full Total column ----------
# All original candidate variables are retained; no outcome-stratified
# columns or association P values are produced.
spec <- list(
  list(section = "Demographic and clinical characteristics", items = list(
    c("Age, years", "age", "num", "0"),
    c("Body mass index, kg/m²", "bmi", "num", "2"),
    c("Sex", "sex", "cat", "Female|Male"),
    c("Smoking", "smoking_status", "cat", "No|Yes"),
    c("Alcohol use", "alcohol_use", "cat", "No|Yes"),
    c("Diabetes", "diabetes", "cat", "No|Yes"),
    c("Hypertension", "hypertension", "cat", "No|Yes"),
    c("Fever", "fever", "cat", "No|Yes"),
    c("Flank pain", "flank_pain", "cat", "No|Yes"),
    c("Prior treatment", "prior_treatment_history", "cat", "No|Yes"),
    c("Hydronephrosis grade", "hydronephrosis_grade", "cat", "Mild|Moderate|Severe"),
    c("Stone side", "stone_side", "cat", "Left|Right"),
    c("Stone location", "stone_location", "cat", "Lower|Middle|Upper"),
    c("Stone number", "stone_multiplicity", "cat", "Single|Multiple")
  )),
  list(section = "Laboratory measures", items = list(
    c("Blood white blood cell count", "blood_wbc_count", "num", "2"),
    c("Urine specific gravity", "urine_specific_gravity", "num", "3"),
    c("Urine pH", "urine_ph", "num", "2"),
    c("Urine white blood cell count", "urine_wbc_count", "num", "2"),
    c("Serum creatinine", "serum_creatinine", "num", "2"),
    c("Blood urea nitrogen", "blood_urea_nitrogen", "num", "2"),
    c("Blood uric acid", "blood_uric_acid", "num", "2")
  )),
  list(section = "CT-derived characteristics", items = list(
    c("Stone attenuation, HU", "stone_attenuation_hu", "num", "2"),
    c("HU proximal to the stone", "hu_above", "num", "2"),
    c("HU distal to the stone", "hu_below", "num", "2"),
    c("Proximal/distal HU ratio", "hu_above_below_ratio", "num", "2"),
    c("Proximal ureter width, mm", "proximal_ureter_width", "num", "2"),
    c("Distal ureter width, mm", "distal_ureter_width", "num", "2"),
    c("Proximal/distal ureter width ratio", "proximal_distal_width_ratio", "num", "2"),
    c("Ureteral wall thickness, mm", "ureteral_wall_thickness", "num", "2"),
    c("Stone length, mm", "stone_length", "num", "2"),
    c("Stone width, mm", "stone_width", "num", "2"),
    c("Stone length/width ratio", "stone_length_width_ratio", "num", "2"),
    c("Hydronephrosis area, mm²", "hydronephrosis_area", "num", "2"),
    c("Ureteral wall area, mm²", "ureteral_wall_area", "num", "2")
  ))
)
all_spec <- unlist(lapply(spec, function(group) group$items), recursive = FALSE)
missing_s1 <- setdiff(vapply(all_spec, `[[`, character(1), 2L), names(development))
if (length(missing_s1)) stop("Development data missing old Table 1 variables: ",
                             paste(missing_s1, collapse = ", "))

rows <- list()
add_row <- function(variable, value, row_type = "data") {
  rows[[length(rows) + 1L]] <<- data.frame(Variable = variable, Value = value,
                                           row_type = row_type)
}
for (group in spec) {
  add_row(group$section, "", "section")
  for (item in group$items) {
    label <- item[[1L]]
    variable <- item[[2L]]
    if (!include_model_variables_in_s1 && variable %in% locked) next
    if (item[[3L]] == "num") {
      values <- numeric_value(development[[variable]], variable)
      add_row(label, fmt_median(values, as.integer(item[[4L]]), nrow(development)))
    } else {
      values <- category_value(development[[variable]],
                               strsplit(item[[4L]], "|", fixed = TRUE)[[1L]], variable)
      add_row(label, "", "category")
      for (level in strsplit(item[[4L]], "|", fixed = TRUE)[[1L]]) {
        add_row(paste0("    ", level), fmt_n(values, level, nrow(development)))
      }
    }
  }
}
table_s1 <- do.call(rbind, rows)
names(table_s1)[2L] <- sprintf("Development cohort (n=%d)", nrow(development))

# ---- Main Table 1: exactly six RF inputs, plus observed outcome -----------
comparison <- list(
  c("Flank pain, yes", "flank_pain", "binary"),
  c("Ureteral wall thickness, mm", "ureteral_wall_thickness", "numeric"),
  c("Stone attenuation, HU", "stone_attenuation_hu", "numeric"),
  c("Proximal ureter width, mm", "proximal_ureter_width", "numeric"),
  c("Stone length, mm", "stone_length", "numeric"),
  c("Hydronephrosis area, mm²", "hydronephrosis_area", "numeric")
)
table_1 <- data.frame(Variable = character(), Development = character(),
                      External = character(), SMD = character())
for (item in comparison) {
  variable <- item[[2L]]
  x <- development[[variable]]
  y <- external[[variable]]
  if (item[[3L]] == "binary") {
    row <- c(item[[1L]], fmt_n(x, "Yes", nrow(development)),
             fmt_n(y, "Yes", nrow(external)),
             standardized_difference(x, y, positive = "Yes"))
  } else {
    row <- c(item[[1L]], fmt_median(x, 2L, nrow(development)),
             fmt_median(y, 2L, nrow(external)), standardized_difference(x, y))
  }
  table_1[nrow(table_1) + 1L, ] <- row
}
table_1[nrow(table_1) + 1L, ] <- c(
  "Impacted stone, yes", fmt_n(development$impacted_stone, "Impacted", nrow(development)),
  fmt_n(external$impacted_stone, "Impacted", nrow(external)), ""
)
names(table_1)[2:3] <- c(sprintf("Development (n=%d)", nrow(development)),
                          sprintf("External (n=%d)", nrow(external)))

# ---- Journal-style Word tables: captions and notes outside table ---------
style_table <- function(d, widths, section_rows = integer()) {
  d$row_type <- NULL
  ft <- flextable::flextable(d)
  ft <- flextable::theme_booktabs(ft)
  ft <- flextable::set_table_properties(
    # Older flextable versions reject repeat_headers inside opts_word.
    # Word normally repeats the table header when it spans pages.
    ft, layout = "fixed", opts_word = list(split = FALSE)
  )
  ft <- flextable::width(ft, j = seq_along(widths), width = widths)
  ft <- flextable::font(ft, fontname = "Arial", part = "all")
  ft <- flextable::fontsize(ft, size = 9, part = "body")
  ft <- flextable::fontsize(ft, size = 9, part = "header")
  ft <- flextable::bold(ft, part = "header")
  ft <- flextable::padding(ft, padding.top = 3, padding.bottom = 3,
                           padding.left = 4, padding.right = 4, part = "all")
  ft <- flextable::valign(ft, valign = "center", part = "all")
  ft <- flextable::align(ft, j = 1L, align = "left", part = "all")
  if (ncol(d) > 1L) {
    ft <- flextable::align(ft, j = 2:ncol(d), align = "center", part = "all")
  }
  if (length(section_rows)) {
    ft <- flextable::bg(ft, i = section_rows, bg = "#EEF2F5", part = "body")
    ft <- flextable::bold(ft, i = section_rows, part = "body")
  }
  ft
}

write_docx <- function(ft, title, note, path) {
  doc <- officer::read_docx()
  doc <- officer::body_add_fpar(
    doc, officer::fpar(officer::ftext(title,
      prop = officer::fp_text(font.family = "Arial", font.size = 10.5,
                              bold = TRUE, color = "#202A32")),
                        fp_p = officer::fp_par(keep_with_next = TRUE))
  )
  doc <- flextable::body_add_flextable(doc, ft)
  doc <- officer::body_add_fpar(
    doc, officer::fpar(officer::ftext(note,
      prop = officer::fp_text(font.family = "Arial", font.size = 8.5,
                              color = "#35424A")))
  )
  print(doc, target = path)
}

ft_s1 <- style_table(table_s1, widths = c(3.75, 2.30),
                     section_rows = which(table_s1$row_type == "section"))
ft_1 <- style_table(table_1, widths = c(2.05, 1.80, 1.80, 0.45))

note_s1 <- paste(
  "Values are median [IQR] or n (%). Percentages use the number with observed data.",
  "Where data are missing, [n=...] gives the available denominator.",
  "This table describes all 545 development patients without a fixed training/test split.",
  "CT, computed tomography; HU, Hounsfield units; IQR, interquartile range."
)
note_1 <- paste(
  "Six predictors of the locked random forest are listed above the observed outcome.",
  "Values are median [IQR] or n (%); [n=...] indicates a reduced available denominator.",
  "SMD is the absolute standardized mean difference: continuous variables use the pooled",
  "standard deviation, and flank pain uses the binary-proportion formula.",
  "No significance tests were performed. HU, Hounsfield units; IQR, interquartile range."
)

write_docx(ft_s1, "Table S1. Characteristics of the overall development cohort",
           note_s1, file.path(output_dir, "Table_S1_development_545.docx"))
write_docx(ft_1, paste0("Table 1. Final model predictors in the development ",
                         "and external validation cohorts"),
           note_1, file.path(output_dir, "Table_1_six_predictors_dev_external.docx"))

# Matching editable data exports; captions and notes are in the Word files.
utils::write.csv(table_s1[setdiff(names(table_s1), "row_type")],
                 file.path(output_dir, "Table_S1_development_545.csv"),
                 row.names = FALSE, na = "")
utils::write.csv(table_1, file.path(output_dir, "Table_1_six_predictors_dev_external.csv"),
                 row.names = FALSE, na = "")
message("Saved manuscript tables in: ", normalizePath(output_dir, winslash = "/"))
