from timeseries import fetch
import yfinance as yf

df = fetch.get_historical_data("GOOG", "2004-08-19", "2004-08-26")

df.to_csv("week1-stock-data.csv")