# =============================================================================
# 01_model_evaluation.R
# Evaluate previously fitted XGBoost models across six-month time-windows
# =============================================================================
#
# IMPORTANT:
# The code supplied for this script originally loaded already-fitted XGBoost
# models. It did not include the original hyperparameter-tuning/model-fitting
# code. This script therefore evaluates saved models; it does not claim to
# reproduce model training or hyperparameter selection.
#
# Required input files:
#   data/dataset_0_to_6.txt, ..., data/dataset_18_to_24.txt
#   models/xgboost_model_0_to_6.json, ..., models/xgboost_model_18_to_24.json
#
# Each dataset must contain:
#   casInd   : binary outcome (1 = pancreatic cancer case; 0 = control)
#   casenum  : matched-set identifier
#   e_patid, e_pracid : identifiers removed before modelling
#   agediag, gender and the predictor variables used by the model
#
# Model files must be compatible with the installed XGBoost version.
# =============================================================================

source("00_project_config.R")
source("00_matched_set_helpers.R")

library(data.table)
library(dplyr)
library(janitor)
library(caret)
library(xgboost)
library(pROC)
library(openxlsx)

# -----------------------------------------------------------------------------
# Helper: preprocess predictors using training-set parameters only
# -----------------------------------------------------------------------------
prepare_predictors <- function(train_data, new_data,
                               outcome = "casInd",
                               id_columns = c("e_patid", "e_pracid", "casenum")) {
  train <- as.data.frame(train_data)
  new   <- as.data.frame(new_data)

  # Standardise age using training-set mean and SD.
  if ("agediag" %in% names(train)) {
    age_mean <- mean(train$agediag, na.rm = TRUE)
    age_sd <- sd(train$agediag, na.rm = TRUE)
    if (!is.finite(age_sd) || age_sd == 0) {
      stop("Training-set age has zero or undefined standard deviation.")
    }
    train$agediag <- (train$agediag - age_mean) / age_sd
    new$agediag <- (new$agediag - age_mean) / age_sd
  }

  # Recode sex consistently. The original analysis used M = 1 and F = 2.
  if ("gender" %in% names(train)) {
    recode_gender <- function(x) {
      x <- as.character(x)
      dplyr::case_when(
        x == "M" ~ 1,
        x == "F" ~ 2,
        TRUE ~ suppressWarnings(as.numeric(x))
      )
    }
    train$gender <- recode_gender(train$gender)
    new$gender <- recode_gender(new$gender)
  }

  remove_cols <- unique(c(id_columns, outcome))
  train_x <- train |> select(-any_of(remove_cols))
  new_x   <- new   |> select(-any_of(remove_cols))

  # Match the training columns and their order exactly.
  missing_in_new <- setdiff(names(train_x), names(new_x))
  extra_in_new <- setdiff(names(new_x), names(train_x))
  if (length(missing_in_new) || length(extra_in_new)) {
    stop(
      "Predictor columns differ between training and test data. Missing: ",
      paste(missing_in_new, collapse = ", "),
      "; extra: ", paste(extra_in_new, collapse = ", ")
    )
  }
  new_x <- new_x[, names(train_x), drop = FALSE]

  # XGBoost requires numeric predictors. Convert factors/characters explicitly.
  non_numeric <- names(train_x)[
    !vapply(train_x, is.numeric, logical(1))
  ]
  if (length(non_numeric)) {
    stop(
      "Non-numeric predictors found: ", paste(non_numeric, collapse = ", "),
      ". Apply the same documented encoding used during model development."
    )
  }

  if (anyNA(train_x) || anyNA(new_x)) {
    stop("Missing predictor values detected. Apply the prespecified missing-data procedure.")
  }

  list(
    train_x = as.matrix(train_x),
    new_x = as.matrix(new_x)
  )
}

# -----------------------------------------------------------------------------
# Helper: calculate performance at a specified classification threshold
# -----------------------------------------------------------------------------
threshold_metrics <- function(observed, predicted_probability, threshold) {
  predicted_class <- as.integer(predicted_probability >= threshold)
  observed <- as.integer(observed)

  tp <- sum(predicted_class == 1 & observed == 1)
  tn <- sum(predicted_class == 0 & observed == 0)
  fp <- sum(predicted_class == 1 & observed == 0)
  fn <- sum(predicted_class == 0 & observed == 1)

  sensitivity <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  specificity <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_

  data.frame(
    threshold = threshold,
    sensitivity = sensitivity,
    specificity = specificity,
    balanced_accuracy = mean(c(sensitivity, specificity), na.rm = TRUE),
    TP = tp, FP = fp, TN = tn, FN = fn
  )
}

# -----------------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------------
all_metrics <- list()
all_threshold_metrics <- list()

for (window_start in WINDOW_STARTS) {
  window_end <- window_start + 6L
  window_name <- paste0(window_start, "_to_", window_end)

  message("Evaluating time-window: ", window_name, " months")

  data_file <- file.path(DATA_DIR, paste0("dataset_", window_name, ".txt"))
  model_file <- file.path(MODEL_DIR, paste0("xgboost_model_", window_name, ".json"))

  if (!file.exists(data_file)) stop("Dataset not found: ", data_file)
  if (!file.exists(model_file)) stop("Model not found: ", model_file)

  dat <- fread(data_file) |> clean_names()

  # clean_names() converts casInd to cas_ind and casenum to casenum.
  if (!all(c("cas_ind", "casenum") %in% names(dat))) {
    stop("Input dataset must contain casInd and casenum columns.")
  }

  split <- split_matched_sets(dat, seed = RANDOM_SEED)

  train_data <- split$train
  test_data <- split$test

  # Check matched-set integrity in both partitions.
  stopifnot(length(intersect(unique(train_data$casenum),
                             unique(test_data$casenum))) == 0)

  processed <- preprocess_predictor_pair(train_data, test_data)
  test_matrix <- processed$x_new
  test_label <- processed$y_new

  # Load model. xgb.load() reads the native XGBoost model format.
  fitted_model <- xgb.load(model_file)

  # Ensure predictor names/order agree with the model when feature names exist.
  model_features <- xgb.feature_names(fitted_model)
  if (!is.null(model_features) && length(model_features) > 0) {
    if (!identical(colnames(test_matrix), model_features)) {
      stop(
        "Predictor names/order do not match the saved model for window ",
        window_name, ". Check preprocessing and feature construction."
      )
    }
  }

  test_dmatrix <- xgb.DMatrix(test_matrix)
  predicted_probability <- predict(fitted_model, test_dmatrix)

  roc_object <- pROC::roc(
    response = test_label,
    predictor = predicted_probability,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )

  auc_value <- as.numeric(pROC::auc(roc_object))
  auc_ci <- as.numeric(pROC::ci.auc(roc_object))

  # Apply the threshold selected using training-set out-of-fold predictions.
  metadata_file <- file.path(MODEL_DIR, paste0("model_metadata_", window_name, ".rds"))
  if (!file.exists(metadata_file)) stop("Model metadata missing: ", metadata_file)
  model_metadata <- readRDS(metadata_file)
  if (is.null(model_metadata$selected_threshold)) {
    stop("Training-derived threshold missing from model metadata. Retrain model.")
  }
  selected_threshold <- model_metadata$selected_threshold
  threshold_results <- threshold_metrics(test_label, predicted_probability,
                                          selected_threshold) |>
    mutate(window = window_name, .before = 1)
  best_threshold <- threshold_results

  all_metrics[[window_name]] <- data.frame(
    window = window_name,
    n_test = length(test_label),
    n_cases = sum(test_label == 1),
    n_controls = sum(test_label == 0),
    auc = auc_value,
    auc_ci_lower = auc_ci[1],
    auc_ci_upper = auc_ci[3],
    selected_threshold = best_threshold$threshold,
    sensitivity = best_threshold$sensitivity,
    specificity = best_threshold$specificity,
    balanced_accuracy = best_threshold$balanced_accuracy
  )

  all_threshold_metrics[[window_name]] <- threshold_results

  # Row-level predictions and matched-set identifiers are deliberately not
  # exported, to avoid disclosure risks under the CPRD data-use agreement.
}

performance_summary <- bind_rows(all_metrics)
threshold_summary <- bind_rows(all_threshold_metrics)

write.csv(
  performance_summary,
  file.path(RESULTS_DIR, "xgboost_performance_by_time_window.csv"),
  row.names = FALSE
)
write.csv(
  threshold_summary,
  file.path(RESULTS_DIR, "xgboost_threshold_metrics_by_time_window.csv"),
  row.names = FALSE
)

message("Model evaluation completed. Results saved in: ", RESULTS_DIR)
