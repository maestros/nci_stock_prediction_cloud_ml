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

# Importing Training Set
start_time = time.time()
###############################
# Setting the timeframe for the data extraction
today = date.today()
date_today = today.strftime("%Y-%m-%d")
date_start = (dt.datetime.now() - timedelta(days=3000)).strftime("%Y-%m-%d")

# Getting YFinance quotes
stockname = 'MSFT'
symbol = 'MSFT'
df = yf.download(symbol, start=date_start, end=date_today)

# Set the sequence length - this is the timeframe used to make a single prediction
sequence_length = 10  # = number of neurons in the first layer of the neural network

# Create a quick overview of the dataset
train_dfs = df.copy()

df = train_dfs.dropna()
################ Data leak check #######

# # Select target
y = train_dfs.Close

# # Select predictors
X = train_dfs.drop(['Close'], axis=1)

y = y.reset_index(drop=True).copy()

train_size = int(len(df) * 0.80)
test_size = len(df) - train_size

##################### RandomForestRegressor check
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from numpy import mean, std

X_trainl, X_testl, y_trainl, y_testl = train_test_split(X, y, test_size=0.4, random_state=0)
model = RandomForestRegressor()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_trainl, y_trainl, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,
                           error_score='raise')

RepeatedKFoldMAE = (mean(n_scores), std(n_scores))

################### Linear regressor check

from sklearn import linear_model
from sklearn import model_selection

train_features, test_features, train_targets, test_targets = model_selection.train_test_split(X_trainl, y_trainl,
                                                                                              test_size=0.2)
# Create an instance of a least-square regression algorithm and assess it's accuracy
# with default hyper-parameter settings
reg = linear_model.LinearRegression()
reg = reg.fit(train_features, train_targets)

from sklearn import metrics

# Predict the response for test dataset
y_pred = reg.predict(test_features)
meanSquaredErrorTestDataAccuracy = metrics.mean_squared_error(test_targets, y_pred)

from sklearn.model_selection import cross_val_score

lm = linear_model.LinearRegression()
scores = cross_val_score(lm, X_testl, y_testl, scoring='neg_mean_squared_error', cv=10)
linearRegressionCrossValidationDataMAE = (mean(scores), std(scores))

#################################################

print("Loading the ML model...")
model = load_model("../deploy/nci_stock_prediction.pkl")
x_test = pickle.load(open("x_test.pkl","rb"))
y_test = pickle.load(open("y_test.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
scaler_pred = pickle.load(open("scaler_pred.pkl","rb"))
print("Loaded the ML model...")

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
# List of Features
FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume']

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
