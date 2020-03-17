import logging


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
