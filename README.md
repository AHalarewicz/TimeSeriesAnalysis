# Time Series Analysis of Historical Stock Data

This projects aims at analysing historical stock data to predict future closing price while exhibiting good software engineering practices.

A Recurrent Neural Network, implemented with keras, will be used to predict whether a stock's closing price will be higher or lower than the closing of the previous day.

A Recurrent Neural Network, implemented with TensorFlow, will be use to make imporved predictions.

# COMMAND LINE INSTRUCTIONS
From the TimeSeries/ directory
1. $ pip install -e .
2. $ pip install -r requirements.txt
3. $ fetch_raw_data --ticker JPM

   Several other ticker symbols that work well with the model include [GE, XOM, BA, GOOG]

4. $ format_timeseries
5. $ predict_tomorrow

    OR fit and predict RNN in TimeSeries/notebooks/LSTM_predictions.ipynb
 
