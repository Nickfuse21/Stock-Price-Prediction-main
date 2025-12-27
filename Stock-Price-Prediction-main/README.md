# Stock-Price-Prediction

Developed a stock price prediction model using ARIMA and LSTMs to forecast future trends. Performed time-series analysis, feature engineering, and model evaluation for accuracy improvement.


here's the code in R

library(quantmod)
library(forecast)
library(keras)
library(tensorflow)
library(tidyverse)

# Load stock data (Example: Apple - AAPL)
getSymbols("AAPL", src = "yahoo", from = "2020-01-01", to = "2024-01-01")
stock_data <- Cl(AAPL)  # Closing Prices

# Convert to time-series
ts_data <- ts(stock_data, frequency = 252)  # 252 trading days in a year

# Split into train-test
train_size <- floor(0.8 * length(ts_data))
train_data <- ts_data[1:train_size]
test_data <- ts_data[(train_size + 1):length(ts_data)]

# 1. ARIMA Model
arima_model <- auto.arima(train_data)
arima_forecast <- forecast(arima_model, h = length(test_data))

# Plot ARIMA results
plot(arima_forecast, main="ARIMA Forecast vs Actual", col="blue")
lines(test_data, col="red")
legend("topright", legend = c("Forecast", "Actual"), col = c("blue", "red"), lty = 1)

# 2. LSTM Model
train_matrix <- as.matrix(train_data)
test_matrix <- as.matrix(test_data)

# Normalize data
train_scaled <- scale(train_matrix)
test_scaled <- scale(test_matrix)

# Convert to supervised learning format
X_train <- array(train_scaled[1:(length(train_scaled)-1)], dim = c(length(train_scaled)-1, 1, 1))
y_train <- train_scaled[2:length(train_scaled)]

# Build LSTM Model
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(1, 1)) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(loss = 'mean_squared_error', optimizer = 'adam')

# Train the model
model %>% fit(X_train, y_train, epochs = 50, batch_size = 1, verbose = 1)

# Prepare test data
X_test <- array(test_scaled[1:(length(test_scaled)-1)], dim = c(length(test_scaled)-1, 1, 1))
y_test <- test_scaled[2:length(test_scaled)]

# Predict with LSTM
lstm_predictions <- model %>% predict(X_test)
lstm_predictions <- lstm_predictions * attr(train_scaled, 'scaled:scale') + attr(train_scaled, 'scaled:center')

# Plot LSTM predictions
plot(test_data, col = "red", main = "LSTM Forecast vs Actual")
lines(lstm_predictions, col = "blue")
legend("topright", legend = c("Forecast", "Actual"), col = c("blue", "red"), lty = 1)

cat("Stock price prediction using ARIMA & LSTM completed.")

