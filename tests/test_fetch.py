from timeseries import fetch
import pandas as pd
import yfinance as yf
from pandas.testing import assert_frame_equal


def test_get_historical_data():
        
        expected = pd.read_csv("validation_data/week1-stock-data.csv")
        
        result = fetch.get_historical_data("GOOG", "2004-08-19", "2004-08-26")
        
        assert type(result)==type(pd.DataFrame())
        
        result.to_csv('test_data/historical.csv')
        
        read = pd.read_csv('test_data/historical.csv')
        
        assert type(result)==type(pd.DataFrame())
        
        assert_frame_equal(expected, read)
        
        