from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from timeseries import bootstrap

file_path = '~/springboard1/capstone2/TimeSeries/data/interim/time_series.csv'
N_LAYERS = 25
N_NODES = 50
TEST_SIZE = 0.25
EPOCHS = 8


N_SAMPLES = 100
accuracy_samples = bootstrap.collect_samples(N_SAMPLES, file_path)

bootstrapping = bootstrap.bootstrap_mean(accuracy_samples)

replicates, int_min, int_max = bootstrapping

bootstrap.save_list(accuracy_samples)

print(bootstrapping)