#from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
#import matplotlib.pyplot as plt

def read_data(file):
    """
    Read csv data from the specified file location.
    """
    df = pd.read_csv(file, index_col='Date')
    return df



def format_predictors_and_targets(df):
    
    df = df.dropna()
    
    predictors = df[['back_5', 'back_4', 'back_3', 'back_2', 'back_1']].values
    assert type(predictors) is np.ndarray
    
    n_cols = predictors.shape[1]
    
    targets = df[['Adj Close']].values
    assert type(targets) is np.ndarray
    
    return predictors, targets, n_cols


def build_sequential(n_nodes, n_layers, n_cols):
    
    model = Sequential()
    
    model.add(Dense(n_nodes, activation='relu', input_shape=(n_cols,)))
    
    for i in range(n_layers-1):
        model.add(Dense(n_nodes, activation='relu'))
    
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


def revert_exp(predictions, y_test):
    predictions = np.exp(predictions)
    y_test_exp = np.exp(y_test)
    return predictions, y_test_exp


def get_accuracy(y, pred):
    
    #scale and shift binary results
    # -1 -> stock went down
    # +1 -> stock increased or stayed the same
    y = ((y>=1)*2)-1
    pred = ((pred>=1)*2)-1
    
    # stocks move in the same direction when a_i*b_i is positive
    accuracy = (np.sum((y*pred)>=0)/len(y))*100
    
    print("Predicting change in stock price with %f%s accuracy" % (accuracy,'%'))
    
    return accuracy