#from keras.models import load_model
import pandas as pd
import numpy as np
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


def build_sequential_Dense(n_nodes, n_layers, n_cols):
    
    model = Sequential()
    
    model.add(Dense(n_nodes, activation='relu', input_shape=(n_cols,)))
    
    for i in range(n_layers-1):
        model.add(Dense(n_nodes, activation='relu'))
    
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


def build_sequential_LSTM(n_nodes, n_layers, add_dense, X_train):
    
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


def revert_exp(predictions, y_test):
    predictions = np.exp(predictions)
    y_test_exp = np.exp(y_test)
    return predictions, y_test_exp


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

