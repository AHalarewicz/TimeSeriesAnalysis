import logging
import yfinance as yf
from datetime import datetime
import pandas as pd

# Define Constants
TICKER = "GOOG"
START = "2004-08-19" # Google IPO date
TODAY = datetime.date(datetime.now()).strftime("%Y-%m-%d")
OUTPUT_FILE_NAME = "raw.csv"
OUTPUT_FILE_PATH = "../../data/raw/" + OUTPUT_FILE_NAME


def get_historical_data(ticker="GOOG", start_date="2004-08-20", end_date=TODAY):
    """
    Collect historical data from yfinance.
    
    Parameters
    ----------
    tick = Ticker for desired stock data
    start_date = startdate of desired stock data
    end_date = end_date of desired stock data)
    
    Returns
    -------
    pandas dataframe with historical data
    columns = stock features
    rows = values of features for individual trading days
    """
    
    # log data collection
    logging.info('Collecting historical stock data from yfinance')
    
    # request data from yfinance
    print('Downloading data from yfinance')
    daily_data = yf.download(ticker, start_date, end_date)
    
    # return data frame with desired data
    return daily_data


# get all current google stock data
stock_data = get_historical_data(TICKER, START, TODAY)

# log population of csv file
logging.info("Writing yfinance data to " + OUTPUT_FILE_PATH)
print("Writing yfinance data to " + OUTPUT_FILE_PATH)

# write data to csv
stock_data.to_csv(OUTPUT_FILE_PATH)




