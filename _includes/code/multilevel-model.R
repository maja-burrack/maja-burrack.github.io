library(readr)
library(dplyr)
library(tidyr)
library(lme4)
library(Metrics)

# Load and transform data
file <- "_data/ifsc_boulder_results_2025.csv"
data <- read_csv(file)

# Drop the first column (assumed to be an index)
data <- data[, -1]

# Convert variables to factors (categorical)
data$comp_id <- as.factor(data$comp_id)
data$event_id <- as.factor(data$event_id)
data$athlete_id <- as.factor(data$athlete_id)
data$athlete_country <- as.factor(data$athlete_country)

# Reshape data: wide format for scores across rounds
unstacked_scores <- data %>%
  select(comp_id, athlete_id, round, score) %>%
  pivot_wider(names_from = round, values_from = score)

# Remove round and score from original data
data <- data %>%
  select(-round, -score) %>%
  distinct()

# Join the reshaped scores
data <- inner_join(data, unstacked_scores, by = c("comp_id", "athlete_id"))

# Rename score columns for clarity
data <- data %>%
  rename(
    score_quali = Qualification,
    score_semi = `Semi-final`,
    score_final = Final
  )

test_comp <- 1408
test_data <- filter(data, event_id == test_comp)
data <- filter(data, event_id != test_comp)

model <- lmer(
    score_final ~ 1 + score_semi + score_quali +  (1 | event_name:gender) + (1 | athlete_name),
    data = data
)

summary(model)

data$pred <- predict(model)
data$residual <- round(data$score_final - data$pred, 1)


# Compute RMSE and MAE
rmse_value <- rmse(data$score_final, data$pred)
mae_value <- mae(data$score_final, data$pred)

# Print
cat("RMSE:", round(rmse_value, 4), "\n")
cat("MAE :", round(mae_value, 4), "\n")

# Select and print the specified columns
print(data[, c("comp_id", "athlete_name", "score_final", "pred", "residual")])

test_data$pred <- predict(model, test_data, allow.new.levels = TRUE)
test_data$residual <- round(test_data$score_final - test_data$pred, 1)

print(test_data[, c("event_name", "gender", "athlete_name", "score_final", "pred", "residual")])
