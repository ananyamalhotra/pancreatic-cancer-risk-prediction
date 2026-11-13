# =============================================================================
# 03_calibration_assessment.R
# Calibration assessment in the held-out matched case-control test sample
# =============================================================================
#
# IMPORTANT INTERPRETATION:
# The study used a matched case-control sample with an approximately 20% case
# fraction. Predicted probabilities from this sample are not population
# absolute risks. This script assesses calibration against the observed
# case-control sample prevalence only.
#
# It does NOT establish population-level calibration. A separate population-
# representative external validation dataset is required for that purpose.
#
# Required inputs:
#   data/dataset_12_to_18.txt
#   models/xgboost_model_12_to_18.json
# =============================================================================

source("00_project_config.R")
source("00_matched_set_helpers.R")

library(data.table)
library(dplyr)
library(janitor)
library(xgboost)
library(pROC)
library(ggplot2)

WINDOW_START <- 12L
WINDOW_END <- 18L
WINDOW_NAME <- paste0(WINDOW_START, "_to_", WINDOW_END)

data_file <- file.path(DATA_DIR, paste0("dataset_", WINDOW_NAME, ".txt"))
model_file <- file.path(MODEL_DIR, paste0("xgboost_model_", WINDOW_NAME, ".json"))

if (!file.exists(data_file)) stop("Dataset not found: ", data_file)
if (!file.exists(model_file)) stop("Model not found: ", model_file)

dat <- fread(data_file) |> clean_names()

# Use the identical case-led matched-set partition and preprocessing used in
# model training and SHAP analysis.
split <- split_matched_sets(dat, seed = RANDOM_SEED)
training <- split$train
testing <- split$test
prepared <- preprocess_predictor_pair(training, testing)
X_test <- prepared$x_new
y_test <- prepared$y_new

model <- xgb.load(model_file)
model_features <- xgb.feature_names(model)
if (!is.null(model_features) && length(model_features) > 0 &&
    !identical(colnames(X_test), model_features)) {
  stop("Predictor names/order do not match saved model.")
}

p_raw <- predict(model, xgb.DMatrix(X_test))
p_raw <- pmin(pmax(p_raw, 1e-8), 1 - 1e-8)
lp_raw <- qlogis(p_raw)

# Calibration-in-the-large: intercept from logistic regression with fixed slope 1.
fit_citl <- glm(y_test ~ 1 + offset(lp_raw), family = binomial())
citl <- unname(coef(fit_citl)[1])

# Calibration intercept and slope: jointly estimated in the test sample.
fit_calibration <- glm(y_test ~ lp_raw, family = binomial())
calibration_intercept <- unname(coef(fit_calibration)[1])
calibration_slope <- unname(coef(fit_calibration)[2])

# Discrimination based on continuous probabilities.
roc_test <- pROC::roc(
  response = y_test,
  predictor = p_raw,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)
auc_test <- as.numeric(pROC::auc(roc_test))
auc_ci <- as.numeric(pROC::ci.auc(roc_test))

calibration_summary <- data.frame(
  n_test = length(y_test),
  observed_case_fraction = mean(y_test),
  mean_predicted_probability = mean(p_raw),
  calibration_in_the_large = citl,
  calibration_intercept = calibration_intercept,
  calibration_slope = calibration_slope,
  auc = auc_test,
  auc_ci_lower = auc_ci[1],
  auc_ci_upper = auc_ci[3]
)

write.csv(
  calibration_summary,
  file.path(RESULTS_DIR, paste0("calibration_summary_", WINDOW_NAME, ".csv")),
  row.names = FALSE
)

# Grouped calibration plot (deciles of predicted probability).
calibration_plot_data <- data.frame(
  predicted = p_raw,
  observed = y_test
) |>
  mutate(group = dplyr::ntile(predicted, 10)) |>
  group_by(group) |>
  summarise(
    mean_predicted = mean(predicted),
    observed_fraction = mean(observed),
    n = n(),
    .groups = "drop"
  )

calibration_plot <- ggplot(
  calibration_plot_data,
  aes(x = mean_predicted, y = observed_fraction)
) +
  geom_point(size = 2) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    x = "Mean predicted probability",
    y = "Observed proportion of cases",
    title = paste0(
      "Calibration in matched case-control test sample (",
      WINDOW_START, "–", WINDOW_END, " months)"
    )
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(RESULTS_DIR, paste0("calibration_plot_", WINDOW_NAME, ".png")),
  calibration_plot,
  width = 7,
  height = 6,
  dpi = 300
)

message(
  "Calibration assessment completed. These results describe calibration ",
  "within the matched case-control test sample, not population absolute risk."
)
