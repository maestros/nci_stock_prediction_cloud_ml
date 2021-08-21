import math  # Mathematical functions
import numpy as np  # Fundamental package for scientific computing with Python
import pandas as pd  # Additional functions for analysing and manipulating data
from datetime import date, timedelta, datetime  # Date Functions
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Packages for measuring model performance / errors
from keras.models import Sequential  # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense, \
    Dropout  # Deep learning classes for recurrent and regular densely-connected layers
from keras.callbacks import EarlyStopping  # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, \
    MinMaxScaler  # This Scaler removes the median and scales the data according to the quantile range to normalize the price data

import yfinance as yf  # Alternative package if webreader does not work: pip install yfinance
from datetime import datetime, timedelta
import time
import datetime as dt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
import json
from keras.models import load_model
import pickle

print("Loading the ML model...")
model = load_model("../deploy/nci_stock_prediction.pkl")
x_test = pickle.load(open("x_test.pkl","rb"))
y_test = pickle.load(open("y_test.pkl","rb"))

print("Loaded the ML model...")

#################################################

# Get the predicted values
y_pred = model.predict(x_test)

# Unscale the predicted values
y_pred_unscaled = scaler_pred.inverse_transform(y_pred)
y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred_unscaled) / y_test_unscaled))) * 100

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred_unscaled) / y_test_unscaled))) * 100

# # Mean Absolute Error (MAE)
MAE = np.mean(abs(y_pred_unscaled - y_test_unscaled))

# Median Absolute Error (MedAE)
MEDAE = np.median(abs(y_pred_unscaled - y_test_unscaled))

from sklearn.metrics import mean_squared_error

RSME = str(np.round(mean_squared_error(y_test_unscaled, y_pred_unscaled, squared=False), 2))

###############################################
### Prediction of the next day price #######
###############################################

df_temp = df[-sequence_length:]
new_df = df_temp.filter(FEATURES)

N = sequence_length

# Get the last N day closing price values and scale the data to be values between 0 and 1
last_N_days = new_df[-sequence_length:].values
last_N_days_scaled = scaler.transform(last_N_days)

# Create an empty list and Append past N days
X_test_new = []
X_test_new.append(last_N_days_scaled)

# Convert the X_test data set to a numpy array and reshape the data
pred_price_scaled = model.predict(np.array(X_test_new))
pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))

# Print last price and predicted price for the next day
price_today = np.round(new_df['Close'][-1], 2)
predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
change_percent = np.round(100 - (price_today * 100) / predicted_price, 2)

plus = '+';
minus = ''
# Calculate prices prediction run execution time
end_time = time.time()
exec_time = end_time - start_time

output_dict = {
"repeatedKFoldMAE": RepeatedKFoldMAE,
"meanSquaredErrorTestDataAccuracy": meanSquaredErrorTestDataAccuracy,
"linearRegressionCrossValidationDataMAE": linearRegressionCrossValidationDataMAE,
"MAPE": MAPE,
"MDAPE": MDAPE,
"MAE": MAE,
"MEDAE": MEDAE,
"RSME": RSME,
"stockname": stockname,
"today": date_today,
"todayClosingPrice": str(price_today),
"predictedPrice": str(predicted_price),
"changePercent": str(change_percent),
"executionTime": str(exec_time)
}

output_json= json.dumps(output_dict)

print(output_json)
