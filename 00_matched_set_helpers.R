# Shared matched-set partition and preprocessing helpers.
# Source this file from every analysis script to guarantee one partition rule.

split_matched_sets <- function(data, outcome = "cas_ind", set_id = "casenum",
                               train_prop = 0.75, seed = 2025L) {
  stopifnot(all(c(outcome, set_id) %in% names(data)))
  cases <- data |> dplyr::filter(.data[[outcome]] == 1)
  controls <- data |> dplyr::filter(.data[[outcome]] == 0)
  eligible_ids <- intersect(unique(cases[[set_id]]), unique(controls[[set_id]]))
  cases <- cases |> dplyr::filter(.data[[set_id]] %in% eligible_ids)
  controls <- controls |> dplyr::filter(.data[[set_id]] %in% eligible_ids)
  if (length(eligible_ids) < 2L) stop("Insufficient complete matched sets.")

  # This is the case-led caret partition used by the training script.
  set.seed(seed)
  selected_case_rows <- caret::createDataPartition(
    cases[[set_id]], p = train_prop, list = FALSE
  )
  train_ids <- unique(cases[[set_id]][selected_case_rows])
  out <- list(
    train = dplyr::bind_rows(
      cases |> dplyr::filter(.data[[set_id]] %in% train_ids),
      controls |> dplyr::filter(.data[[set_id]] %in% train_ids)
    ),
    test = dplyr::bind_rows(
      cases |> dplyr::filter(!(.data[[set_id]] %in% train_ids)),
      controls |> dplyr::filter(!(.data[[set_id]] %in% train_ids))
    ),
    train_set_ids = train_ids,
    test_set_ids = setdiff(eligible_ids, train_ids)
  )
  stopifnot(length(intersect(out$train_set_ids, out$test_set_ids)) == 0L)
  out
}

preprocess_predictor_pair <- function(train_data, new_data,
                                      outcome = "cas_ind",
                                      id_columns = c("e_patid", "e_pracid", "casenum")) {
  tr <- as.data.frame(train_data); nw <- as.data.frame(new_data)
  age_mean <- mean(tr$agediag, na.rm = TRUE)
  age_sd <- stats::sd(tr$agediag, na.rm = TRUE)
  if (!is.finite(age_sd) || age_sd == 0) stop("Invalid training-set age SD.")
  tr$agediag <- (tr$agediag - age_mean) / age_sd
  nw$agediag <- (nw$agediag - age_mean) / age_sd
  recode_gender <- function(x) {
    x <- as.character(x)
    dplyr::case_when(x == "M" ~ 1, x == "F" ~ 2,
                     TRUE ~ suppressWarnings(as.numeric(x)))
  }
  tr$gender <- recode_gender(tr$gender); nw$gender <- recode_gender(nw$gender)
  y_train <- as.integer(tr[[outcome]]); y_new <- as.integer(nw[[outcome]])
  drop <- c(id_columns, outcome)
  x_train <- tr |> dplyr::select(-dplyr::any_of(drop))
  x_new <- nw |> dplyr::select(-dplyr::any_of(drop))
  if (!identical(names(x_train), names(x_new))) stop("Predictor columns/order differ.")
  if (!all(vapply(x_train, is.numeric, logical(1)))) stop("All predictors must be numeric.")
  if (anyNA(x_train) || anyNA(x_new)) stop("Missing predictors found; apply the prespecified missing-data method.")
  list(x_train = as.matrix(x_train), y_train = y_train,
       x_new = as.matrix(x_new), y_new = y_new,
       age_mean = age_mean, age_sd = age_sd)
}
