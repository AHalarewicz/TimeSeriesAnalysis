import logging
#logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)
import yfinance as yf
from datetime import datetime
import pandas as pd
#from timeseries import data
from timeseries import fetch

# Define Constants
#TICKER = "GOOG"
TICKER = "XOM"
START = "1900-01-01" # Google IPO date
TODAY = datetime.date(datetime.now()).strftime("%Y-%m-%d")
OUTPUT_FILE_NAME = "raw.csv"
#OUTPUT_FILE_PATH = "../../data/raw/" + OUTPUT_FILE_NAME
OUTPUT_FILE_PATH = "~/springboard1/capstone2/TimeSeries/data/raw/" + OUTPUT_FILE_NAME

# get all current google stock data
#stock_data = data.get_historical_data(TICKER, START, TODAY)
stock_data = fetch.get_historical_data(TICKER, START, TODAY)

# log population of csv file
logging.info("Writing yfinance data to " + OUTPUT_FILE_PATH)
print("Writing yfinance data to " + OUTPUT_FILE_PATH)

# write data to csv
stock_data.to_csv(OUTPUT_FILE_PATH)
