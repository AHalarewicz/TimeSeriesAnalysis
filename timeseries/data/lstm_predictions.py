import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def read_data(file):
    """
    Read csv data from the specified file location.
    """
    df = pd.read_csv(file, index_col='Date')
    return df

def clip_recent_days(df, n_days):
    """
    -- optional --
    remove recent days from data frame
    """
    return(df[:-n_days])


def format_predictors_and_targets(df):
    
    """
    params: DataFrame with Predictors and Targets
    
    drop empty rows and prepare targets and predictors
    
    returns: formatted predictors, targets, and number of predictor columns
    
    """
    
    #drop rows with missing values
    df = df.dropna()
    
    # Extract Predictors
    predictors = df[['back_5', 'back_4', 'back_3', 'back_2', 'back_1']].values
    assert type(predictors) is np.ndarray
    
    # count number of predictive columns
    n_cols = predictors.shape[1]
    
    # extract target
    targets = df[['Adj Close']].values
    assert type(targets) is np.ndarray
    
    return predictors, targets, n_cols



def build_sequential_LSTM(n_nodes, n_layers, add_dense):
    
    """
    Params: number of nodes in layers, number of LSTM layers, option to add Dense Layer
    
    Build a Sequential keras model with LSTM and optional Dense Layers
    
    Returns: Keras model with LSTM layers with minimum of 2 LSTM layers
    
    """
    
    # create Sequential model object
    model = Sequential()
    
    # add initial LSTM Layer
    model.add(LSTM(n_nodes, return_sequences=True, input_shape=X_train.shape[1:]))
    

    # add specified amount of additional LSTM layers
    for i in range(n_layers-2):
        model.add(LSTM(n_nodes, return_sequences=True))
    
    #return_sequences = False if next layer is not LSTM
    model.add(LSTM(n_nodes, return_sequences=False))
    
    
    if add_dense:
        # add optional Fully Connected Layer
        model.add(Dense(n_nodes, activation='relu'))
        
    # add Dense layer to produce output
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


def get_accuracy(y, pred):
    
    """
    params: targets from the test set, model predictions
    
    Scores the model's ability to correctly predict the direction of change
    Magnitude of predicted changes is not scored.
    
    returns: model accuracy against the test set
    """
    #scale and shift binary results
    # -1 -> stock went down
    # +1 -> stock increased or stayed the same
    y = ((y>=1)*2)-1
    pred = ((pred>=1)*2)-1
    
    # stocks move in the same direction when a_i*b_i is positive
    accuracy = (np.sum((y*pred)>=0)/len(y))*100
    
    print("Predicting change in stock price with %f%s accuracy" % (accuracy,'%'))
    
    return accuracy




# PREPARE DATA
TEST_SIZE = 0.05
file_path = '~/springboard1/capstone2/TimeSeries/data/interim/time_series.csv'

# read time series data
time_series_df = read_data(file_path)


# extract last row to predict tomorrow's change
tomorrow = time_series_df.iloc[[-1]].fillna(0)

# call function to format predictors and targets
tomorrow, _ , _ = format_predictors_and_targets(tomorrow)
predictors, targets, n_cols = format_predictors_and_targets(time_series_df)


# scale data to range [0,1]

# create scaler objects
X_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

# fit respective scalers to data
predictors = X_scaler.fit_transform(predictors)
tomorrow = X_scaler.transform(tomorrow)
targets = y_scaler.fit_transform(targets)


# test for correct scaling
assert min(predictors.flatten()) == 0
assert max(predictors.flatten()) == 1
assert min(targets.flatten()) == 0
assert max(targets.flatten()) == 1


# split data into training set and testing set
# SHUFFLE = FALSE
X_train, X_test, y_train, y_test = train_test_split(predictors, targets, test_size=TEST_SIZE, shuffle=False, stratify=None, random_state=1)

# test for sequential split
assert np.argwhere(predictors == X_train[-1])[0][0] == (np.argwhere(predictors == X_test[0])[0][0]) -1
assert np.argwhere(predictors == y_train[-1])[0][0] == (np.argwhere(predictors == y_test[0])[0][0]) -1


# BUILD MODEL

# re-shape predictors for keras model
tomorrow = np.reshape(tomorrow, (1, 1, tomorrow.shape[1]))
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

assert X_train.shape[1:] == tomorrow.shape[1:]

# DEFINE MODEL CONSTANTS
N_NODES = 100
N_LAYERS = 4
ADD_DENSE = True


# build Sequential Model
model = build_sequential_LSTM(N_NODES, N_LAYERS, ADD_DENSE)


# Fit the Model Exclusively with Training Data

# DEFINE TRAINING CONSTANTS
EPOCHS = 2

# fit model to training data
model.fit(X_train, y_train, epochs=EPOCHS)


# save and load model
model.save('../models/keras_lstm.h5')
model = load_model('../models/keras_lstm.h5')


# Make Predictions and interpret results

# Make Predictions
predictions = model.predict(X_test)
tomorrows_prediction = model.predict(tomorrow)

# revert scaling
tomorrow_unscaled = y_scaler.inverse_transform(tomorrows_prediction)
unscaled_predictions = y_scaler.inverse_transform(predictions)
unscaled_y_test = y_scaler.inverse_transform(y_test)

# apply exponential function
exponential_tomorrow = np.exp(tomorrow_unscaled)
exponential_predictions = np.exp(unscaled_predictions)
exponential_y_test = np.exp(unscaled_y_test)

# Inspect Quality of Predictions
# Inspect quality of predictions
places = 4
min_pred = round(float(min(exponential_predictions)), places)
max_pred = round(float(max(exponential_predictions)), places)
mean_pred = round(float(np.mean(exponential_predictions)), places)
median_pred = round(float(np.median(exponential_predictions)), places)
percentile = round(np.percentile(exponential_predictions, 1.0), places)*100
print("min pred:\t", min_pred)
print("max pred:\t", max_pred)
print("mean pred:\t", mean_pred)
print("median pred:\t", median_pred)
print("percentile(1.0):", percentile)


is_good_model = min_pred<=1.0 and max_pred>=1.0
assert is_good_model


# SCORE MODEL

# get accuracy
accuracy = get_accuracy(exponential_y_test, exponential_predictions)


# TOMORROW'S PREDICTIONS

def action(x):
    """
    params: expected change in stock price
    
    Map to expected changes in the action column
    
    Return: "Buy" if expected increase, "SELL" if expected decrease
    """
    if x>0:
        return "BUY"
    else:
        return "SELL"
    
# read original Adj Close Data
# read data with Adj Close Price
raw_data = pd.read_csv('~/springboard1/capstone2/TimeSeries/data/raw/raw.csv', index_col=['Date'])

# display a dataFrame providing insight to th user
expected_return = float(exponential_tomorrow)
tomorrow_df = raw_data[['Adj Close']].iloc[[-1]]
tomorrow_df.columns = ['price_today']
tomorrow_df['price_tomorrow'] = tomorrow_df['price_today']* expected_return
tomorrow_df['Action'] = (tomorrow_df.price_tomorrow - tomorrow_df.price_today).map(action)
print(tomorrow_df)


