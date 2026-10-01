# Figure 2 | Nested algorithm comparison and selection
# Replots both panels directly from impacted_nested_lasso_9models.R results.
# Run this AFTER the original nested analysis has finished.

project_dir <- "C:/Users/Lenovo/Desktop/impacted_pr"
output_dir <- file.path(project_dir, "output", "nested_lasso_9models")

# The original script writes to fully_nested_lasso_9models by default, but
# accepts a different output directory as its second command-line argument.
result_candidates <- file.path(
  project_dir, "output",
  c("nested_lasso_9models", "fully_nested_lasso_9models"),
  "fully_nested_lasso_9models_results.rds"
)
available <- result_candidates[file.exists(result_candidates)]
if (length(available) != 1L) {
  stop(
    "Expected exactly one results RDS. Found ", length(available), ".\n",
    "Set `results_file` explicitly if both folders contain an RDS.\n",
    paste(result_candidates, collapse = "\n")
  )
}
results_file <- available[[1L]]

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Install ggplot2 first: install.packages('ggplot2')")
}
library(ggplot2)
library(grid)

res <- readRDS(results_file)
needed <- c("all_model_fold_metrics", "algorithm_selection_frequency",
            "model_ranking", "selected_model", "settings")
if (!all(needed %in% names(res))) {
  stop("The RDS does not have the expected nested-analysis result fields.")
}

auc <- as.data.frame(res$all_model_fold_metrics)
freq <- as.data.frame(res$algorithm_selection_frequency)
ranking <- as.data.frame(res$model_ranking)
models <- as.character(ranking$model)
labels <- setNames(as.character(freq$model_label), as.character(freq$model))

outer_v <- as.integer(res$settings$value[res$settings$setting == "outer_v"])
outer_repeats <- as.integer(
  res$settings$value[res$settings$setting == "outer_repeats"]
)
n_outer <- outer_v * outer_repeats

if (length(models) != 9L || anyDuplicated(models) ||
    !setequal(auc$model, models) || !setequal(freq$model, models) ||
    anyNA(labels[models]) || length(n_outer) != 1L || is.na(n_outer)) {
  stop("Nine-model identifiers or the outer-CV configuration are inconsistent.")
}
if (anyNA(auc$roc_auc) || any(auc$roc_auc < 0 | auc$roc_auc > 1) ||
    anyDuplicated(auc[c("outer_id", "model")]) ||
    !all(table(factor(auc$model, levels = models)) == n_outer)) {
  stop("The outer-fold AUC data are incomplete or invalid.")
}
if (anyNA(freq$selected_outer_folds) ||
    sum(freq$selected_outer_folds) != n_outer ||
    anyDuplicated(freq$model) || !res$selected_model %in% models) {
  stop("Algorithm selection counts do not match the outer folds.")
}

# Identical model order in both panels: the final selection ranking.
auc$model <- factor(auc$model, levels = rev(models))
freq$model <- factor(freq$model, levels = rev(models))
auc$selected <- as.character(auc$model) == res$selected_model
freq$selected <- as.character(freq$model) == res$selected_model

colors <- c(`FALSE` = "#397B9D", `TRUE` = "#B65C32")
fills <- c(`FALSE` = "#DCEAF2", `TRUE` = "#F6D9C7")

auc_min <- max(0.5, floor((min(auc$roc_auc) - 0.01) * 20) / 20)
auc_max <- min(1, ceiling((max(auc$roc_auc) + 0.01) * 20) / 20)
auc_ticks <- seq(auc_min, auc_max, by = 0.05)
count_max <- ceiling((max(freq$selected_outer_folds) + 2) / 2) * 2

common_theme <- theme_classic(base_size = 10, base_family = "sans") +
  theme(
    plot.title = element_text(size = 11, face = "bold", margin = margin(b = 7)),
    plot.title.position = "plot",
    axis.title.x = element_text(size = 10, margin = margin(t = 7)),
    axis.text = element_text(size = 9, color = "#27343C"),
    axis.line = element_line(linewidth = 0.4, color = "#27343C"),
    axis.ticks = element_line(linewidth = 0.4, color = "#27343C"),
    plot.margin = margin(7, 12, 9, 5)
  )

panel_a <- ggplot(auc, aes(x = roc_auc, y = model)) +
  geom_boxplot(
    aes(fill = selected), width = 0.60, outlier.shape = NA,
    color = "#24506A", linewidth = 0.40
  ) +
  geom_point(
    aes(color = selected),
    position = position_jitter(width = 0, height = 0.11, seed = 2026),
    size = 1.15, alpha = 0.72
  ) +
  scale_fill_manual(values = fills, guide = "none") +
  scale_color_manual(values = colors, guide = "none") +
  scale_x_continuous(breaks = auc_ticks, labels = function(x) sprintf("%.2f", x)) +
  scale_y_discrete(labels = function(x) unname(labels[x])) +
  coord_cartesian(xlim = c(auc_min, auc_max), clip = "off") +
  labs(title = "A   Outer-fold discrimination", x = "Outer-fold ROC AUC", y = NULL) +
  common_theme

panel_b <- ggplot(freq, aes(x = selected_outer_folds, y = model)) +
  geom_col(aes(fill = selected), width = 0.61) +
  geom_text(
    aes(label = selected_outer_folds),
    hjust = 0, nudge_x = 0.18, size = 3.1, color = "#27343C"
  ) +
  scale_fill_manual(values = colors, guide = "none") +
  scale_x_continuous(
    limits = c(0, count_max), breaks = seq(0, count_max, by = 2),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_discrete(labels = function(x) unname(labels[x])) +
  labs(title = "B   Algorithm selection frequency",
       x = sprintf("Outer training sets selected (out of %d)", n_outer),
       y = NULL) +
  common_theme

# Align the plotting areas and keep native vector text and graphics in the PDF.
grobs <- list(ggplotGrob(panel_a), ggplotGrob(panel_b))
shared_widths <- unit.pmax(grobs[[1]]$widths, grobs[[2]]$widths)
grobs[[1]]$widths <- shared_widths
grobs[[2]]$widths <- shared_widths

draw_figure <- function() {
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(2, 1, heights = unit(c(1, 1), "null"))))
  grid.draw(editGrob(grobs[[1]], vp = viewport(layout.pos.row = 1)))
  grid.draw(editGrob(grobs[[2]], vp = viewport(layout.pos.row = 2)))
  popViewport()
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
file_base <- file.path(output_dir, "Figure_2_nested_algorithm_selection")

if (capabilities("cairo")) {
  cairo_pdf(paste0(file_base, ".pdf"), width = 7.2, height = 8.6,
            family = "sans", bg = "white")
} else {
  pdf(paste0(file_base, ".pdf"), width = 7.2, height = 8.6,
      family = "sans", bg = "white")
}
draw_figure()
dev.off()

tiff(paste0(file_base, ".tiff"), width = 7.2, height = 8.6,
     units = "in", res = 600, compression = "lzw", bg = "white")
draw_figure()
dev.off()

message("Source: ", results_file)
message("Saved: ", file_base, ".pdf")
message("Saved: ", file_base, ".tiff")
