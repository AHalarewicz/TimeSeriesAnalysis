import sys
import logging
import click
from timeseries import pipelines

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

# if you want to provide a filepath as a command line argument
#@click.command()
#@click.option('--filename',
#              type=click.Path(exists=True),
#              prompt='Path to the Google Stock CSV file',
#              help='Path to the Google Stock CSV file')
@click.command()
@click.option('-t', '--ticker', 'ticker')

def fetch_raw_data(ticker):
    # collect historical stock data for the specified ticker
    pipelines.run_fetch_raw_data(ticker)
    
def format_timeseries():
    # format historical stock data into Time Series
    pipelines.run_format_timeseries()
    
def predict_tomorrow():
    # train the model and predict tomorrow's change
    pipelines.run_predict_tomorrow()
