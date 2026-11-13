# =============================================================================
# 02_shap_analysis.R
# SHAP feature importance and summary plots for each prediction window
# =============================================================================
#
# This script explains previously fitted XGBoost models using SHAP values.
# It does not refit the models.
#
# Required input files:
#   data/dataset_<start>_to_<end>.txt
#   models/xgboost_model_<start>_to_<end>.json
#   metadata/feature_labels.xlsx
#
# The analysis dataset must have the same predictor definitions and coding
# used to develop the corresponding model.
# =============================================================================

source("00_project_config.R")
source("00_matched_set_helpers.R")

library(data.table)
library(dplyr)
library(janitor)
library(caret)
library(xgboost)
library(shapviz)
library(ggplot2)
library(patchwork)
library(readxl)

# Read optional human-readable feature labels.
label_file <- file.path(METADATA_DIR, "feature_labels.xlsx")
if (!file.exists(label_file)) {
  stop("Feature label dictionary not found: ", label_file)
}
feature_labels <- read_excel(label_file) |>
  rename_with(tolower)

if (!all(c("variable", "label") %in% names(feature_labels))) {
  stop("feature_labels.xlsx must contain columns named 'variable' and 'label'.")
}
label_dictionary <- setNames(feature_labels$label, feature_labels$variable)

for (window_start in WINDOW_STARTS) {
  window_end <- window_start + 6L
  window_name <- paste0(window_start, "_to_", window_end)
  message("Generating SHAP plots for window: ", window_name)

  data_file <- file.path(DATA_DIR, paste0("dataset_", window_name, ".txt"))
  model_file <- file.path(MODEL_DIR, paste0("xgboost_model_", window_name, ".json"))

  if (!file.exists(data_file) || !file.exists(model_file)) {
    warning("Skipping missing dataset/model for window ", window_name)
    next
  }

  dat <- fread(data_file) |> clean_names()
  split <- split_matched_sets(dat, seed = RANDOM_SEED)

  # Explain the held-out test observations, not the training observations.
  matrices <- preprocess_predictor_pair(split$train, split$test)
  X_explain <- matrices$x_new

  model <- xgb.load(model_file)
  model_features <- xgb.feature_names(model)

  if (!is.null(model_features) && length(model_features) > 0 &&
      !identical(colnames(X_explain), model_features)) {
    stop("Feature names/order do not match model for window ", window_name)
  }

  shap_object <- shapviz(model, X = X_explain)

  # Mean absolute SHAP importance.
  importance_data <- sv_importance(shap_object, kind = "bar", show_numbers = TRUE)$data

  # Apply readable labels where available; otherwise retain feature names.
  importance_data$display_label <- ifelse(
    importance_data$feature %in% names(label_dictionary),
    unname(label_dictionary[importance_data$feature]),
    importance_data$feature
  )

  importance_plot <- ggplot(
    importance_data,
    aes(x = reorder(display_label, value), y = value)
  ) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(
      x = NULL,
      y = "Mean absolute SHAP value",
      title = "A. Mean absolute feature importance"
    ) +
    theme_minimal(base_size = 12)

  # SHAP beeswarm summary. shapviz uses feature names from the model matrix.
  beeswarm_plot <- sv_importance(shap_object, kind = "beeswarm") +
    ggtitle("B. SHAP summary") +
    theme_minimal(base_size = 12)

  # Rename y-axis feature labels if the package version supports the scale.
  beeswarm_plot <- beeswarm_plot +
    scale_y_discrete(labels = function(x) {
      out <- unname(label_dictionary[x])
      out[is.na(out)] <- x[is.na(out)]
      out
    })

  combined_plot <- (importance_plot + beeswarm_plot) +
    plot_layout(ncol = 2) +
    plot_annotation(
      title = paste0(
        "SHAP analysis: ", window_start, "–", window_end,
        " months before diagnosis"
      ),
      theme = theme(
        plot.title = element_text(size = 15, face = "bold", hjust = 0.5)
      )
    )

  ggsave(
    filename = file.path(
      RESULTS_DIR,
      paste0("shap_summary_", window_name, ".png")
    ),
    plot = combined_plot,
    width = 12,
    height = 9,
    dpi = 300
  )

  # Save numeric mean(|SHAP|) values as a machine-readable supplementary output.
  write.csv(
    importance_data,
    file.path(RESULTS_DIR, paste0("shap_importance_", window_name, ".csv")),
    row.names = FALSE
  )
}

message("SHAP analysis completed.")
