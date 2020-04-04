import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from timeseries import wrangle
from timeseries import fetch



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
    
