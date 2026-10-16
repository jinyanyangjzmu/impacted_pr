# Figure 2: nested algorithm comparison and selection
#
# Replots saved results; does not fit models or change model selection.
# Dependency: ggplot2 >= 3.4.0. The original project path is preserved below.
# You may still supply paths explicitly:
#   Rscript code/Figure_2_from_nested_results.R [results.rds] [output_dir]
#   Rscript code/Figure_2_from_nested_results.R --help
#
# Expected RDS fields:
#   all_model_fold_metrics: model, outer_id, roc_auc
#   algorithm_selection_frequency: model, model_label, selected_outer_folds
#   model_ranking: model (in the upstream selection order)
#   selected_model: one model identifier
#   settings: setting, value (including outer_v and outer_repeats)
#
# Upstream analysis requirements: impacted stone is the positive event;
# AUROC uses its predicted probability with a fixed direction. Algorithm
# selection uses inner resampling within each outer-training set. This
# plotting script cannot verify these requirements from summary metrics.

# ---- Input validation --------------------------------------------------------

require_columns <- function(data, columns, name) {
  missing <- setdiff(columns, names(data))
  if (length(missing)) {
    stop(name, " is missing: ", paste(missing, collapse = ", "), call. = FALSE)
  }
}

read_cv_setting <- function(settings, key) {
  value <- suppressWarnings(as.numeric(as.character(
    settings$value[settings$setting == key]
  )))
  if (length(value) != 1L || !is.finite(value) ||
      value < 1 || value != floor(value)) {
    stop("Invalid CV setting: ", key, call. = FALSE)
  }
  value
}

project_dir <- "C:/Users/Lenovo/Desktop/impacted_pr"
default_output_dir <- file.path(project_dir, "output", "nested_lasso_9models")

find_results <- function(project_dir) {
  standard_candidates <- file.path(
    project_dir, "output",
    c("nested_lasso_9models", "fully_nested_lasso_9models"),
    "fully_nested_lasso_9models_results.rds"
  )
  found <- standard_candidates[file.exists(standard_candidates)]
  if (length(found) == 1L) return(found[[1L]])

  # The analysis script may use a different output folder or RDS filename.
  # Search the original project output directory without changing that path.
  output_root <- file.path(project_dir, "output")
  all_rds <- if (dir.exists(output_root)) {
    list.files(output_root, pattern = "\\.rds$", recursive = TRUE,
               full.names = TRUE, ignore.case = TRUE)
  } else {
    character()
  }
  preferred <- all_rds[grepl("nested|lasso|model", basename(all_rds),
                             ignore.case = TRUE)]
  if (length(preferred) == 1L) return(preferred[[1L]])
  if (length(all_rds) == 1L) return(all_rds[[1L]])

  if (interactive()) {
    message("The standard RDS filename was not found. Please select the nested-analysis RDS file.")
    selected <- tryCatch(file.choose(), error = function(e) "")
    if (nzchar(selected) && file.exists(selected)) return(selected)
  }

  searched <- c(standard_candidates, all_rds)
  stop(
    "No unique nested-analysis RDS file was found.\n",
    "The plotting script needs the saved result object, not the R script itself.\n",
    "Expected standard locations:\n", paste(standard_candidates, collapse = "\n"),
    if (length(all_rds)) paste0("\n\nRDS files found under output/:\n",
                                paste(all_rds, collapse = "\n")) else
      "\n\nNo .rds files were found under the project output/ directory.",
    "\nPass the actual RDS path as the first command-line argument or run this script interactively and select it.",
    call. = FALSE
  )
}

# ---- Figure generation -------------------------------------------------------

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (any(args %in% c("-h", "--help"))) {
    cat("Usage: Rscript Figure_2_from_nested_results.R [results.rds] [output_dir]\n",
        "With no arguments, searches the original project output folders.\n",
        "Default project folder: C:/Users/Lenovo/Desktop/impacted_pr\n",
        "Exports: vector PDF, 600-dpi TIFF, and figure legend TXT.\n", sep = "")
    return(invisible(NULL))
  }
  if (length(args) > 2L) {
    stop("Supply at most two arguments: results.rds and output_dir.", call. = FALSE)
  }
  if (!requireNamespace("ggplot2", quietly = TRUE) ||
      utils::packageVersion("ggplot2") < "3.4.0") {
    stop("Install ggplot2 >= 3.4.0: install.packages('ggplot2')", call. = FALSE)
  }
  library(ggplot2)

  results_file <- if (length(args)) args[[1L]] else find_results(project_dir)
  results_file <- normalizePath(results_file, winslash = "/", mustWork = TRUE)
  output_dir <- if (length(args) == 2L) args[[2L]] else default_output_dir
  res <- readRDS(results_file)
  fields <- c("all_model_fold_metrics", "algorithm_selection_frequency",
              "model_ranking", "selected_model", "settings")
  if (!is.list(res) || !all(fields %in% names(res))) {
    stop("Results RDS does not match the documented schema.", call. = FALSE)
  }

  auc <- as.data.frame(res$all_model_fold_metrics)
  freq <- as.data.frame(res$algorithm_selection_frequency)
  ranking <- as.data.frame(res$model_ranking)
  settings <- as.data.frame(res$settings)
  require_columns(auc, c("model", "outer_id", "roc_auc"), "Fold metrics")
  require_columns(freq, c("model", "model_label", "selected_outer_folds"),
                  "Selection frequencies")
  require_columns(ranking, "model", "Model ranking")
  require_columns(settings, c("setting", "value"), "Settings")

  models <- as.character(ranking$model)
  selected_model <- as.character(res$selected_model)
  auc$model <- as.character(auc$model)
  freq$model <- as.character(freq$model)
  outer_v <- read_cv_setting(settings, "outer_v")
  outer_repeats <- read_cv_setting(settings, "outer_repeats")
  n_outer <- outer_v * outer_repeats

  if (length(models) != 9L || anyNA(models) || any(!nzchar(models)) ||
      anyDuplicated(models) || !setequal(auc$model, models) ||
      !setequal(freq$model, models) || anyDuplicated(freq$model) ||
      length(selected_model) != 1L || is.na(selected_model) ||
      !selected_model %in% models) {
    stop("Expected nine consistent model identifiers and one selected model.",
         call. = FALSE)
  }
  labels <- setNames(as.character(freq$model_label), freq$model)
  if (anyNA(labels) || any(!nzchar(trimws(labels)))) {
    stop("Model display labels must be non-missing and non-empty.", call. = FALSE)
  }
  if (!is.numeric(auc$roc_auc) || any(!is.finite(auc$roc_auc)) ||
      any(auc$roc_auc < 0 | auc$roc_auc > 1) || anyNA(auc$outer_id) ||
      anyDuplicated(auc[c("outer_id", "model")]) ||
      length(unique(auc$outer_id)) != n_outer ||
      !all(table(factor(auc$model, levels = models)) == n_outer)) {
    stop("AUROC must be valid for every model in the same outer resamples.",
         call. = FALSE)
  }
  counts <- freq$selected_outer_folds
  if (!is.numeric(counts) || any(!is.finite(counts)) || any(counts < 0) ||
      any(counts != floor(counts)) || sum(counts) != n_outer ||
      counts[match(selected_model, freq$model)] != max(counts)) {
    stop("Selection counts must sum to n_outer; the selected model must have\n",
         "the highest count (ties allowed under the upstream selection rule).",
         call. = FALSE)
  }
  selected_count <- counts[match(selected_model, freq$model)]

  # Use the upstream ranking in both panels; do not rank by assessment AUROC.
  auc$model <- factor(auc$model, levels = rev(models))
  freq$model <- factor(freq$model, levels = rev(models))
  auc$selected <- as.character(auc$model) == selected_model
  freq$selected <- as.character(freq$model) == selected_model
  colors <- c(`FALSE` = "#397B9D", `TRUE` = "#B65C32")
  fills <- c(`FALSE` = "#DFEDF4", `TRUE` = "#F5DED0")
  auc_range <- c(max(0, floor((min(auc$roc_auc) - 0.01) * 20) / 20),
                 min(1, ceiling((max(auc$roc_auc) + 0.01) * 20) / 20))
  count_step <- max(1, ceiling(max(counts) / 5))
  count_max <- ceiling((max(counts) + count_step) / count_step) * count_step

  figure_theme <- theme_classic(base_size = 11, base_family = "sans") +
    theme(
      plot.title = element_text(size = 12, face = "bold", margin = margin(b = 3)),
      plot.subtitle = element_text(size = 10, margin = margin(b = 8)),
      plot.title.position = "plot",
      axis.title.x = element_text(size = 11, margin = margin(t = 8)),
      axis.text = element_text(size = 10, color = "#27343C"),
      axis.line = element_line(linewidth = 0.4, color = "#27343C"),
      axis.ticks = element_line(linewidth = 0.4, color = "#27343C"),
      plot.margin = margin(8, 14, 10, 6)
    )

  panel_a <- ggplot(auc, aes(x = roc_auc, y = model)) +
    geom_boxplot(aes(fill = selected), orientation = "y", width = 0.58,
                 outlier.shape = NA, coef = 1.5,
                 color = "#24506A", linewidth = 0.4) +
    geom_point(aes(color = selected), size = 1.3, alpha = 0.75,
               position = position_jitter(width = 0, height = 0.11, seed = 2026)) +
    scale_fill_manual(values = fills, guide = "none") +
    scale_color_manual(values = colors, guide = "none") +
    scale_x_continuous(breaks = seq(auc_range[1], auc_range[2], by = 0.05),
                       labels = function(x) sprintf("%.2f", x)) +
    scale_y_discrete(labels = function(x) unname(labels[x])) +
    coord_cartesian(xlim = auc_range) +
    labs(title = "A   Outer-assessment AUROC",
         subtitle = sprintf("%d resamples: %d repeats of %d-fold outer CV",
                            n_outer, outer_repeats, outer_v),
         x = "AUROC", y = NULL) +
    figure_theme

  panel_b <- ggplot(freq, aes(x = selected_outer_folds, y = model)) +
    geom_col(aes(fill = selected), width = 0.60, orientation = "y") +
    geom_text(aes(label = selected_outer_folds), hjust = 0,
              nudge_x = count_step * 0.09, size = 3.5, color = "#27343C") +
    scale_fill_manual(values = colors, guide = "none") +
    scale_x_continuous(limits = c(0, count_max),
                       breaks = seq(0, count_max, by = count_step),
                       expand = expansion(mult = c(0, 0))) +
    scale_y_discrete(labels = function(x) unname(labels[x])) +
    labs(title = "B   Algorithm selection frequency",
         subtitle = sprintf("Selection within %d outer-training sets", n_outer),
         x = sprintf("Number of selections (out of %d)", n_outer), y = NULL) +
    figure_theme +
    theme(
      # Panel A supplies the shared algorithm labels in the horizontal layout.
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank()
    )

  # Align panel heights and place the two panels side by side.
  grobs <- list(ggplotGrob(panel_a), ggplotGrob(panel_b))
  heights <- grid::unit.pmax(grobs[[1]]$heights, grobs[[2]]$heights)
  grobs[[1]]$heights <- grobs[[2]]$heights <- heights
  draw_figure <- function() {
    grid::grid.newpage()
    grid::pushViewport(
      grid::viewport(
        layout = grid::grid.layout(
          1, 2, widths = grid::unit(c(1.12, 0.88), "null")
        )
      )
    )
    for (i in seq_along(grobs)) {
      grid::grid.draw(grid::editGrob(
        grobs[[i]], vp = grid::viewport(layout.pos.col = i)
      ))
    }
    grid::popViewport()
  }

  # Horizontal two-panel export. Retain vector PDF text and high-resolution TIFF.
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  file_base <- file.path(output_dir, "Figure_2_nested_algorithm_selection")
  if (capabilities("cairo")) {
    grDevices::cairo_pdf(paste0(file_base, ".pdf"), width = 10.5, height = 5.6,
                        family = "sans", bg = "white")
  } else {
    grDevices::pdf(paste0(file_base, ".pdf"), width = 10.5, height = 5.6,
                  family = "Helvetica", bg = "white", useDingbats = FALSE)
  }
  tryCatch(draw_figure(), finally = grDevices::dev.off())
  grDevices::tiff(paste0(file_base, ".tiff"), width = 10.5, height = 5.6,
                 units = "in", res = 600, compression = "lzw", bg = "white")
  tryCatch(draw_figure(), finally = grDevices::dev.off())

  # A separate legend preserves readable figure labels at publication size.
  legend <- paste0(
    "Figure 2. Nested comparison and selection of nine candidate algorithms. ",
    "(A) Each point represents AUROC in one outer-assessment resample from ",
    outer_repeats, " repeats of ", outer_v, "-fold outer cross-validation (", 
    n_outer, " resamples per algorithm). Boxes show the median and ",
    "interquartile range; whiskers extend to the most extreme observations ",
    "within 1.5 interquartile ranges of the box. All resample values are shown. ",
    "The repeated resamples share patients and are not independent observations; ",
    "these distributions are descriptive, not confidence intervals or pooled ",
    "patient-level OOF AUROCs. (B) Selection counts from inner resampling within ",
    "each outer-training set. Outer-assessment performance was not used for ",
    "these training-stage selections. The locked algorithm, ",
    labels[[selected_model]], ", was selected in ",
    selected_count, " of ", n_outer,
    " outer-training sets and is highlighted in orange. This highlight indicates ",
    "final selection, not statistically established superiority. Impacted stone ",
    "is the positive event in the upstream analysis. AUROC, area under the ",
    "receiver operating characteristic curve; CV, cross-validation; OOF, ",
    "out-of-fold."
  )
  writeLines(strwrap(legend, width = 88), paste0(file_base, "_legend.txt"))
  message("Source: ", results_file)
  message("Saved PDF, TIFF, and legend: ", file_base)
  invisible(list(panel_a = panel_a, panel_b = panel_b, legend = legend))
}

main()
