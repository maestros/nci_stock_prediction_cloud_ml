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

import pickle
 
print("Serializing the ML model...")
model.save("../deploy/nci_stock_prediction.pkl")
pickle.dump(x_test, open("../deploy/x_test.pkl","wb"))
pickle.dump(y_test, open("../deploy/y_test.pkl","wb"))
pickle.dump(scaler, open("../deploy/scaler.pkl","wb"))
pickle.dump(scaler_pred, open("../deploy/scaler_pred.pkl","wb"))
print("Finished serializing the ML model: ../deploy/nci_stock_prediction.pkl")

