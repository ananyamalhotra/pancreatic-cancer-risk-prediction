# =============================================================================
# 01_model_training_and_evaluation.R
# Train XGBoost models and evaluate performance for each time-window
# =============================================================================
#
# This script implements the tuning strategy documented in the supplied
# analysis code. It uses a 75:25 matched-set split and five-fold CV within the
# training sample. All members of a matched set remain in the same fold.
#
# CAUTION:
# The parameter grid below contains 1,728 combinations. This is computationally
# intensive.
#
# The classification threshold is selected using grouped out-of-fold
# predictions from the training set, then evaluated once in the held-out
# test set
# =============================================================================

source("00_project_config.R")
source("00_matched_set_helpers.R")

library(data.table)
library(dplyr)
library(janitor)
library(caret)
library(xgboost)
library(pROC)

# -----------------------------------------------------------------------------
# Data preparation helpers
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Matched-set cross-validation folds for xgb.cv()
# -----------------------------------------------------------------------------
make_matched_folds <- function(set_id, k = 5, seed = 2025L) {
  unique_sets <- unique(set_id)
  set.seed(seed)
  set_folds <- caret::createFolds(unique_sets, k = k, list = TRUE)

  # xgb.cv expects validation row indices, not matched-set indices.
  lapply(set_folds, function(fold_set_indices) {
    validation_set_ids <- unique_sets[fold_set_indices]
    which(set_id %in% validation_set_ids)
  })
}

# -----------------------------------------------------------------------------
# Threshold performance metrics
# -----------------------------------------------------------------------------
calculate_threshold_metrics <- function(y, p, threshold) {
  pred <- as.integer(p >= threshold)
  tp <- sum(pred == 1 & y == 1)
  tn <- sum(pred == 0 & y == 0)
  fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)

  sensitivity <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  specificity <- ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)

  data.frame(
    threshold = threshold,
    sensitivity = sensitivity,
    specificity = specificity,
    balanced_accuracy = (sensitivity + specificity) / 2,
    TP = tp, FP = fp, TN = tn, FN = fn
  )
}

# -----------------------------------------------------------------------------
# Hyperparameter grid: retained from the supplied analysis
# -----------------------------------------------------------------------------
parameter_grid <- expand.grid(
  eta = c(0.05, 0.1, 0.2),
  max_depth = c(4, 6, 8, 10),
  min_child_weight = c(1, 2),
  subsample = c(0.8, 1),
  colsample_bytree = c(0.8, 1),
  gamma = c(0, 0.1, 0.5),
  lambda = c(1, 5, 10),
  alpha = c(0, 0.1)
)

base_parameters <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = 4
)

# -----------------------------------------------------------------------------
# Main loop across time-windows
# -----------------------------------------------------------------------------
performance_by_window <- list()
threshold_results_by_window <- list()

for (window_start in WINDOW_STARTS) {
  window_end <- window_start + 6L
  window_name <- paste0(window_start, "_to_", window_end)
  message("Training model for window: ", window_name, " months")

  data_file <- file.path(DATA_DIR, paste0("dataset_", window_name, ".txt"))
  if (!file.exists(data_file)) stop("Dataset not found: ", data_file)

  dat <- fread(data_file) |> janitor::clean_names()

  required_columns <- c("cas_ind", "casenum", "agediag", "gender")
  if (!all(required_columns %in% names(dat))) {
    stop("Dataset is missing required columns: ",
         paste(setdiff(required_columns, names(dat)), collapse = ", "))
  }

  split <- split_matched_sets(dat, seed = RANDOM_SEED)
  training <- split$train
  testing <- split$test

  # Confirm that no matched set appears in both partitions.
  if (length(intersect(unique(training$casenum), unique(testing$casenum))) > 0) {
    stop("Matched-set leakage detected between training and test partitions.")
  }

  prepared <- preprocess_predictor_pair(training, testing)
  dtrain <- xgb.DMatrix(prepared$x_train, label = prepared$y_train)
  dtest <- xgb.DMatrix(prepared$x_new, label = prepared$y_new)

  # Cross-validation folds are defined by matched set to keep matched members
  # together. The folds are fixed across all parameter combinations.
  matched_folds <- make_matched_folds(
    training$casenum, k = 5, seed = RANDOM_SEED
  )

  tuning_results <- vector("list", nrow(parameter_grid))

  for (grid_row in seq_len(nrow(parameter_grid))) {
    current_parameters <- c(
      base_parameters,
      as.list(parameter_grid[grid_row, , drop = FALSE])
    )

    cv_fit <- xgb.cv(
      params = current_parameters,
      data = dtrain,
      nrounds = 1000,
      folds = matched_folds,
      early_stopping_rounds = 50,
      verbose = 0,
      seed = RANDOM_SEED
    )

    cv_log <- as.data.frame(cv_fit$evaluation_log)
    best_iteration <- cv_fit$best_iteration
    if (is.null(best_iteration) || is.na(best_iteration)) {
      best_iteration <- which.max(cv_log$test_auc_mean)
    }

    tuning_results[[grid_row]] <- list(
      parameters = current_parameters,
      best_iteration = best_iteration,
      best_cv_auc = cv_log$test_auc_mean[best_iteration]
    )
  }

  tuning_table <- data.frame(
    grid_row = seq_along(tuning_results),
    cv_auc = vapply(tuning_results, `[[`, numeric(1), "best_cv_auc"),
    best_iteration = vapply(tuning_results, `[[`, numeric(1), "best_iteration")
  )

  best_row <- which.max(tuning_table$cv_auc)
  best_parameters <- tuning_results[[best_row]]$parameters
  best_iteration <- tuning_results[[best_row]]$best_iteration

  # Refit the selected model using the full training partition.
  final_model <- xgb.train(
    params = best_parameters,
    data = dtrain,
    nrounds = best_iteration,
    verbose = 0
  )

  # Save model in JSON format and save tuning metadata.
  model_output <- file.path(
    MODEL_DIR, paste0("xgboost_model_", window_name, ".json")
  )
  dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)
  xgb.save(final_model, model_output)

  saveRDS(
    list(
      parameters = best_parameters,
      best_iteration = best_iteration,
      best_cv_auc = tuning_table$cv_auc[best_row],
      age_mean = prepared$age_mean,
      age_sd = prepared$age_sd,
      predictor_names = colnames(prepared$x_train),
      seed = RANDOM_SEED
    ),
    file.path(MODEL_DIR, paste0("model_metadata_", window_name, ".rds"))
  )

  # Test-set discrimination using continuous probabilities.
  p_test <- predict(final_model, dtest)
  roc_object <- pROC::roc(
    response = prepared$y_new,
    predictor = p_test,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  auc_value <- as.numeric(pROC::auc(roc_object))
  auc_ci <- as.numeric(pROC::ci.auc(roc_object))

  # Obtain out-of-fold training predictions with the selected hyperparameters.
  # Threshold selection is therefore independent of the held-out test outcomes.
  oof_fit <- xgb.cv(
    params = best_parameters, data = dtrain, nrounds = best_iteration,
    folds = matched_folds, prediction = TRUE, verbose = 0,
    seed = RANDOM_SEED
  )
  p_oof <- oof_fit$pred
  if (is.null(p_oof) || length(p_oof) != length(prepared$y_train)) {
    stop("Could not obtain aligned out-of-fold predictions for threshold selection.")
  }
  threshold_table <- bind_rows(lapply(
    seq(0.10, 0.50, by = 0.01),
    function(threshold) calculate_threshold_metrics(prepared$y_train, p_oof, threshold)
  ))

  best_threshold <- threshold_table |>
    filter(balanced_accuracy == max(balanced_accuracy, na.rm = TRUE)) |>
    arrange(desc(sensitivity), threshold) |>
    slice(1)

  # Store the training-derived threshold alongside the model metadata.
  model_metadata <- readRDS(file.path(MODEL_DIR, paste0("model_metadata_", window_name, ".rds")))
  model_metadata$selected_threshold <- best_threshold$threshold
  saveRDS(model_metadata, file.path(MODEL_DIR, paste0("model_metadata_", window_name, ".rds")))

  threshold_table$window <- window_name
  threshold_results_by_window[[window_name]] <- threshold_table

  performance_by_window[[window_name]] <- data.frame(
    window = window_name,
    n_training = nrow(training),
    n_test = nrow(testing),
    n_test_cases = sum(prepared$y_new == 1),
    n_test_controls = sum(prepared$y_new == 0),
    best_cv_auc = tuning_table$cv_auc[best_row],
    best_iteration = best_iteration,
    test_auc = auc_value,
    test_auc_ci_lower = auc_ci[1],
    test_auc_ci_upper = auc_ci[3],
    selected_threshold = best_threshold$threshold,
    sensitivity = best_threshold$sensitivity,
    specificity = best_threshold$specificity,
    balanced_accuracy = best_threshold$balanced_accuracy
  )

  write.csv(
    tuning_table,
    file.path(RESULTS_DIR, paste0("tuning_results_", window_name, ".csv")),
    row.names = FALSE
  )
}

write.csv(
  bind_rows(performance_by_window),
  file.path(RESULTS_DIR, "xgboost_performance_by_time_window.csv"),
  row.names = FALSE
)
write.csv(
  bind_rows(threshold_results_by_window),
  file.path(RESULTS_DIR, "xgboost_threshold_metrics_by_time_window.csv"),
  row.names = FALSE
)

message("Training and evaluation completed.")
