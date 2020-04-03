import logging
import pandas as pd
from timeseries import fetch
from datetime import datetime


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
    
    
