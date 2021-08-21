# https://github.com/flo7up/relataly-public-python-tutorials/blob/master/.ipynb_checkpoints/012%20Stock%20Market%20Prediction%20using%20Multivariate%20Time%20Series%20Models%20and%20Python-checkpoint.ipynb

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
train_dfs
df = train_dfs.dropna()
# df = train_dfs
################ Data leak check #######

# # Select target
y = train_dfs.Close

# # Select predictors
X = train_dfs.drop(['Close'], axis=1)

y = y.reset_index(drop=True).copy()

train_size = int(len(df) * 0.80)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

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
# scores

# Create the training and test data

# Indexing Batches
train_df = train_dfs.sort_values(by=['Date']).copy()

# We safe a copy of the dates index, before we need to reset it to numbers
date_index = train_df.index

# We reset the index, so we can convert the date-index to a number-index
train_df = train_df.reset_index(drop=True).copy()

################################

# List of Features
FEATURES = ['High', 'Low', 'Open', 'Close', 'Volume']

# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[FEATURES]

# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['Close']

# Get the number of rows in the data
# This  bit helps to resolve data leaks issues by using MinMaxscaler
nrows = data_filtered.shape[0]

# Convert the data to numpy values
np_data_unscaled = np.array(data_filtered)
np_data = np.reshape(np_data_unscaled, (nrows, -1))

# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(data_filtered_ext['Close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

###############################

# Prediction Index
index_Close = 0

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 85% of the data
train_data_len = math.ceil(np_data_scaled.shape[0] * 0.80)

# Create the training and test data
train_data = np_data_scaled[0:train_data_len, :]
test_data = np_data_scaled[train_data_len - sequence_length:, :]


# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i - sequence_length:i, :])  # contains sequence_length values 0-sequence_length * columsn
        y.append(data[
                     i, index_Close])  # contains the prediction values for validation (3rd column = Close),  for single-step prediction

    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y


# Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)

# Validate that the prediction value and the input match up
# The last close price of the second input sample should equal the first prediction value
##########################################

# Configure the neural network model
model = Sequential()

# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]

model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(n_neurons, return_sequences=False))
model.add(Dense(5))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='Adamax', loss='mse')

####################################################

# Training the model
epochs = 1
batch_size = 16
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test)
                    )

############################################


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