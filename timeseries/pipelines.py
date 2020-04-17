import logging
import pandas as pd
import numpy as np
from datetime import datetime
import keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from timeseries import wrangle
from timeseries import fetch
from timeseries import BuildModel



def run_fetch_raw_data(ticker='GOOG'):
    
    """ 
    Get raw data from yfinance and 
    """

    logging.info('Fetching raw data from yfinance')
    
    
    # Define Constants
    #TICKER = "GOOG"
    TICKER = ticker
    #START = "2004-08-19" # Google IPO date
    START = "1900-01-01"
    TODAY = datetime.date(datetime.now()).strftime("%Y-%m-%d")
    OUTPUT_FILE_NAME = "raw.csv"
    OUTPUT_FILE_PATH = "~/springboard1/capstone2/TimeSeries/data/raw/" + OUTPUT_FILE_NAME

    # get all current google stock data
    stock_data = fetch.get_historical_data(TICKER, START, TODAY)

    # log population of csv file
    logging.info("Writing yfinance data to " + OUTPUT_FILE_PATH)
    print("Writing yfinance data to " + OUTPUT_FILE_PATH)

    # write data to csv
    stock_data.to_csv(OUTPUT_FILE_PATH)
    
    return
    
def run_format_timeseries():
    print('formatting timeseries')
    # raw data file path
    RAW_DATA_FILE_PATH = "~/springboard1/capstone2/TimeSeries/data/raw/raw.csv"

    # read csv data from
    raw_df = pd.read_csv(RAW_DATA_FILE_PATH, index_col=0, parse_dates=['Date'])


    adj_close_df = raw_df.iloc[:,4:5]


    log_scaled_adj_close = wrangle.take_log(adj_close_df)
    log_scaled_adj_close, deltas = wrangle.get_deltas(log_scaled_adj_close)

    #previous_df = wrangle.create_previous_days(log_scaled_adj_close, 'Adj Close')

    def create_time_series(df, col_name='Adj Close'):
        df = wrangle.create_previous_days(df, col_name)
        #df = create_future_days(df, col_name)
        return df

    time_series_df = create_time_series(log_scaled_adj_close, 'Adj Close')
    time_series_df.to_csv('~/springboard1/capstone2/TimeSeries/data/interim/time_series.csv')
    #time_series_df.to_csv('../data/interim/time_series.csv')
    print('DONE')


def run_predict_tomorrow():
    # PREPARE DATA
    TEST_SIZE = 0.05
    file_path = '~/springboard1/capstone2/TimeSeries/data/interim/time_series.csv'

    # read time series data
    time_series_df = BuildModel.read_data(file_path)


    # extract last row to predict tomorrow's change
    tomorrow = time_series_df.iloc[[-1]].fillna(0)

    # call function to format predictors and targets
    tomorrow, _ , _ = BuildModel.format_predictors_and_targets(tomorrow)
    predictors, targets, n_cols = BuildModel.format_predictors_and_targets(time_series_df)


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
    model = BuildModel.build_sequential_LSTM(N_NODES, N_LAYERS, ADD_DENSE, X_train)


    # Fit the Model Exclusively with Training Data

    # DEFINE TRAINING CONSTANTS
    EPOCHS = 2

    # fit model to training data
    model.fit(X_train, y_train, epochs=EPOCHS)


    # save and load model
    save_model = False
    if save_model:
        model.save('~/springboard1/capstone2/TimeSeries/models/keras_lstm.h5')
        model = load_model('~/springboard1/capstone2/TimeSeries/models/keras_lstm.h5')


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
    accuracy = BuildModel.get_accuracy(exponential_y_test, exponential_predictions)


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

