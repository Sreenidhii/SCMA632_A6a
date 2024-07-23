# Install Necessary Libraries
install.packages(c("quantmod", "tensorflow", "reticulate", "ggplot2", "forecast", "lubridate", "devtools", "caret", "plotly", "keras", "randomForest"))
devtools::install_github("business-science/tidyquant")

# Load Required Libraries
library(tidyquant)
library(ggplot2)
library(forecast)
library(keras)
library(tensorflow)
library(randomForest)
library(caret)
library(dplyr)
library(lubridate)
library(tensorflow)
library(reticulate)

# 1. Data Fetching from Yahoo Finance
ticker <- "HDFCBANK.NS"
data <- tq_get(ticker, from = "2021-04-01", to = "2024-03-31")
head(data)

# 2. Select the Target Variable and Clean the Data
df <- data[, c("date", "adjusted")]
colnames(df) <- c("Date", "Adj_Close")

# 2.1 Plot the Time Series
ggplot(df, aes(x = Date, y = Adj_Close)) +
  geom_line() +
  ggtitle("HDFCBANK.NS Adj Close Price") +
  xlab("Date") +
  ylab("Adj Close Price") +
  theme_minimal()

library(plotly)
fig <- plot_ly(df, x = ~Date, y = ~Adj_Close, type = 'scatter', mode = 'lines')
fig <- fig %>% layout(title = 'HDFCBANK.NS Adj Close Price',
                      xaxis = list(title = 'Date'),
                      yaxis = list(title = 'Adj Close'))
fig

# 2.2 Decomposition of Time Series
ts_data <- ts(df$Adj_Close, frequency = 12)
decomp <- decompose(ts_data, type = "multiplicative")

# Plot the decomposed components
plot(decomp)

# 3. Split the Data into Training and Test Sets
monthly_data <- df %>%
  mutate(Date = as.Date(Date)) %>%
  group_by(Date = as.yearmon(Date)) %>%
  summarize(Adj_Close = mean(Adj_Close))

train_size <- floor(0.8 * nrow(monthly_data))
train_data <- monthly_data[1:train_size,]
test_data <- monthly_data[(train_size + 1):nrow(monthly_data),]


# 3.1 Holt-Winters Model

# Ensure 'Date' column is in Date format and 'Adj_Close' column is numeric
monthly_data <- monthly_data %>%
  mutate(Date = as.Date(Date)) %>%
  arrange(Date)

# Convert to time series object (assuming monthly data, so frequency = 12)
ts_data <- ts(monthly_data$Adj_Close, start = c(year(min(monthly_data$Date)), month(min(monthly_data$Date))), frequency = 12)

# Check if we have enough data points
if (length(ts_data) < 24) { # For example, 2 years of monthly data
  stop("Not enough data for Holt-Winters model.")
}

# Fit the Holt-Winters model
hw_model <- HoltWinters(ts_data)

# Forecast for the next 12 periods (months)
hw_forecast <- forecast(hw_model, h = 12)

# Plot the forecast
plot(hw_forecast)
lines(ts_data, col = "Blue")
legend("topright", legend = c("Forecast", "Observed"), col = c("Black", "Blue"), lty = 1:1)


# 3.2 ARIMA Monthly Data
train_ts <- ts(train_data$Adj_Close, frequency = 12)
arima_model <- auto.arima(train_ts, seasonal = TRUE, stepwise = TRUE)
forecast_arima <- forecast(arima_model, h = 8)
test_ts <- ts(test_data$Adj_Close, frequency = 12)
autoplot(forecast_arima) + autolayer(test_ts, series="Test Data")


# 3.3 ARIMA Daily Data
daily_data <- ts(df$Adj_Close, frequency = 7)
arima_model_daily <- auto.arima(daily_data, seasonal = TRUE, stepwise = TRUE)
forecast_arima_daily <- forecast(arima_model_daily, h = 60)
autoplot(forecast_arima_daily) + autolayer(daily_data, series="Daily Data")

library(magrittr)
library(keras)


# Scale the data
scaled_data <- scale(df$Adj_Close)

# Create sequences
sequences <- embed(scaled_data, 30)


# Split data into training and testing sets
train_size <- floor(0.8 * nrow(sequences))
x_train <- sequences[1:train_size, 1:30]
y_train <- sequences[2:(train_size + 1), 1]
x_test <- sequences[(train_size + 1):nrow(sequences), 1:30]
y_test <- sequences[(train_size + 1):nrow(sequences), 1]


# 4. Tree-Based Models
# Create sequences
create_sequences <- function(data, target_col, seq_length) {
  num_samples <- nrow(data) - seq_length
  num_features <- ncol(data)
  sequences <- array(NA, dim = c(num_samples, seq_length, num_features))
  labels <- numeric(num_samples)
  
  for (i in 1:num_samples) {
    sequences[i, , ] <- as.matrix(data[i:(i + seq_length - 1), ])
    labels[i] <- data[i + seq_length, target_col]
  }
  
  list(sequences = sequences, labels = labels)
}

sequences_result <- create_sequences(df, 2, 30)
train_size <- floor(0.8 * length(sequences_result$labels))
x_train <- sequences_result$sequences[1:train_size, , ]
y_train <- sequences_result$labels[1:train_size]
x_test <- sequences_result$sequences[(train_size + 1):length(sequences_result$labels), , ]
y_test <- sequences_result$labels[(train_size + 1):length(sequences_result$labels)]

# Ensure 'y' is a numeric vector
y <- as.numeric(sequences_result$labels)

# Flatten the sequences if needed
x <- sequences_result$sequences
n_samples <- dim(x)[1]
seq_length <- dim(x)[2]
n_features <- dim(x)[3]

# Check if y is numeric or factor
str(y)

sequences <- create_sequences(df, target_col = 2, seq_length = 30)
x <- sequences$sequences
y <- sequences$labels

# Extract the numeric vector from the labels component
y_numeric <- as.numeric(y)

# Create the training index
train_index <- createDataPartition(y_numeric, p = 0.8, list = FALSE)


# Verify the dimensions
dim(x_train)
dim(x_test)
length(y_train)
length(y_test)


# Inspect the structure and dimensions of x_train
str(x_train)

# Get the number of features
n_features <- ncol(x_train)
print(n_features)


# Convert the x_train matrix to a data frame with column names
x_train_df <- as.data.frame(x_train)
colnames(x_train_df) <- paste0("X", 1:n_features)

# Check the number of rows and columns in x_train_df and y_train
nrow(x_train_df)
ncol(x_train_df)
nrow(y_train)

# Check if y_train has the same number of elements as x_train_df
if (nrow(x_train_df) != length(y_train)) {
  if (!all(is.na(y_train))) {
    stop("The number of rows in x_train_df and y_train is different.")
  } else {
    warning("The number of rows in y_train is NA, assuming it has the same number of rows as x_train_df.")
  }
}

# If the number of rows is different, subset the input data to match the target variable
if (nrow(x_train_df)!= length(y_train)) {
  x_train_df <- x_train_df[1:length(y_train), ]
}

# Check column names of x_train_df and fix if needed
colnames(x_train_df) <- make.names(colnames(x_train_df), unique = TRUE)



# Check the dimensions of x_train_df and y_train_numeric
cat("Dimensions of x_train_df:", dim(x_train_df), "\n")


# Print a summary of the data
cat("Summary of x_train_df:\n")
print(summary(x_train_df))


# Convert x_train to a data frame if it's not
x_train_df <- as.data.frame(x_train)

# Ensure y_train is a numeric vector
y_train_numeric <- as.numeric(y_train)

# Check for NA values in x_train_df and y_train_numeric
if (any(is.na(x_train_df))) {
  stop("x_train_df contains NA values.")
}

if (any(is.na(y_train_numeric))) {
  stop("y_train_numeric contains NA values.")
}

# Train the Random Forest model
rf_model <- randomForest(x_train_df, y_train_numeric)

# Print a summary of the model
print(rf_model)


# Ensure y_test is a numeric vector
y_test_numeric <- as.numeric(y_test)

# Check if y_test_numeric is indeed numeric
if (!is.numeric(y_test_numeric)) {
  stop("y_test must be a numeric vector.")
}

# Convert x_test to a data frame if it's not already
x_test_df <- as.data.frame(x_test)

# Check and align column names of x_test_df with x_train_df
if (!all(names(x_train_df) %in% names(x_test_df))) {
  stop("x_test does not contain the same feature set as x_train.")
}

# Ensure the columns are in the same order
x_test_df <- x_test_df[, names(x_train_df), drop = FALSE]

# Generate predictions using the Random Forest model
rf_predictions <- predict(rf_model, x_test_df)

# Print the first few predictions
print(head(rf_predictions))


# Plot the true values and the predictions
plot(
  y_test_numeric,
  type = "l",
  col = "Blue",
  main = "Random Forest: Predictions vs True Values",
  ylab = "Adj Close",
  xlab = "Index"
)
lines(rf_predictions, col = "Red")

