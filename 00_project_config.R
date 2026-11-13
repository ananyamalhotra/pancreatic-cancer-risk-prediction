# =============================================================================
# Common project configuration
# =============================================================================
# Set PROJECT_DIR to the directory containing the folders "data", "models",
# "metadata" and "results". 
#
# Expected project layout:
# project/
#   data/                 # locally held, authorised study datasets (not public)
#   models/               # fitted XGBoost model files (may be restricted)
#   metadata/             # feature-label dictionary (non-identifiable)
#   results/              # generated outputs
#
# The scripts assume one row per person per time-window and that casenum
# identifies a matched set (one case and its matched controls).

PROJECT_DIR <- "path/to/your/project"

DATA_DIR    <- file.path(PROJECT_DIR, "data")
MODEL_DIR   <- file.path(PROJECT_DIR, "models")
METADATA_DIR<- file.path(PROJECT_DIR, "metadata")
RESULTS_DIR <- file.path(PROJECT_DIR, "results")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# Required packages. Install them once if needed.
required_packages <- c(
  "data.table", "dplyr", "janitor", "caret", "xgboost",
  "pROC", "ggplot2", "openxlsx", "readxl", "shapviz",
  "patchwork"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Please install the following packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

# Reproducibility seed used for matched-set partitioning.
RANDOM_SEED <- 2025L

# time-windows are six-month windows starting 0, 1, ..., 18 months before PC diagnosis date (index date)
WINDOW_STARTS <- 0:18
