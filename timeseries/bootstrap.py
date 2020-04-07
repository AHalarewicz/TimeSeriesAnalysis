import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import csv

from timeseries import data
from timeseries import BuildModel

TEST_SIZE = 0.25
N_LAYERS = 25
N_NODES = 50
EPOCHS = 8
#mean_accuracy_file = '~/springboard1/capstone2/TimeSeries/data/interim/mean_accuracies.csv'
mean_accuracy_file = '../../data/interim/mean_accuracies.csv'

def read_list():
    with open(mean_accuracy_file, 'r') as file:
        data = csv.reader(file)
        accuracies = []
        for row in data:
            accuracies.append(float(row[0]))
        return accuracies
        
    
def save_list(mylist):
    with open(mean_accuracy_file, 'w') as file:
        for acc in mylist:
            file.write(str(acc))
            file.write('\n')
        

def percentile_p(arr, p):
    """return a tuple of the lower and upper bounds of a p_% confindence interval"""
    ends = 100 - p
    left = ends/2
    right = 100 - left
    return np.percentile(arr, [left, right])


def bootstrap_replicate_1d(data, func):
    """Draw a single bootstrap replicate"""
    return func(np.random.choice(data, size=len(data)))


def draw_bs_reps(data, func, size=1):
    """Draw many bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates



def bootstrap_mean(samples):
    
    """ given a sample, use Bootstrap Statistics to return the distribution and the 95 % confidence interval for the mean as a tuple"""
    # compute observed mean of sample
    observed_mean = np.mean(samples)
    
    # generate 10,000 bootstrap replicates
    N_reps = 100000
    bs_replicates = draw_bs_reps(samples, np.mean, N_reps)
    
    # compute standard error of the mean 
    sem = np.std(samples) / np.sqrt(len(samples))
    
    # compute extremes of 95 percentile
    int_min, int_max = percentile_p(bs_replicates, 95)
    
    conf_min, conf_max = percentile_p(bs_replicates, 95)
    conf_range = bs_replicates[(bs_replicates >= conf_min) & (bs_replicates <= conf_max)]
    
    
    # plot distribution of bootstrap replicates
    _ = plt.figure(figsize=(10,8))
    _ , bins, _ = plt.hist(bs_replicates, bins=50, density=True, alpha=0.5)
    _ = plt.hist(conf_range, bins=bins, density=True, alpha=1, color='b')
    _ = plt.xlabel('mean')
    _ = plt.ylabel('PDF')


    _ = plt.axvline(conf_min, color='w', linestyle='-', linewidth=2.5)
    _ = plt.axvline(conf_max, color='w', linestyle='-', linewidth=2.5)
    _ = plt.axvline(np.mean(bs_replicates), color='w', linestyle=':')
    _ = plt.axvline(observed_mean, color='r', alpha=0.5, linestyle=':')

    _ = plt.title('95% Confidence Interval of the Mean')
    print('Bootstrap Mean:\t\t\t',np.mean(bs_replicates).round(3), '\nObserved Mean of samples: \t',observed_mean.round(3))
    #plt.savefig("~/springboard1/capstone2/TimeSeries/figures/")
    plt.savefig('bootstrap_mean.png')
    plt.show()
    
    return (bs_replicates, int_min, int_max)



def collect_samples(num_samples=100, file_path = '../../data/interim/time_series.csv'):
    
    time_series_df = data.read_data(file_path)
    
    predictors, targets, n_cols = BuildModel.format_predictors_and_targets(time_series_df)
    
    X_scaler = MinMaxScaler(feature_range=(0,1))
    y_scaler = MinMaxScaler(feature_range=(0,1))
    
    predictors = X_scaler.fit_transform(predictors)
    targets = y_scaler.fit_transform(targets)
    
    model = BuildModel.build_sequential(N_NODES, N_LAYERS, n_cols)
    
    samples = []
    
    for i in range(num_samples):
        print('collecting sample: ', i+1)
        
        X_train, X_test, y_train, y_test = train_test_split(predictors, targets, test_size=TEST_SIZE)

        model.fit(X_train, y_train, use_multiprocessing=True, epochs=EPOCHS)
        predictions = model.predict(X_test)

        predictions = y_scaler.inverse_transform(predictions)
        y_test = y_scaler.inverse_transform(y_test)

        predictions, y_test_exp = BuildModel.revert_exp(predictions, y_test)

        accuracy = BuildModel.get_accuracy(y_test_exp, predictions)
        
        samples.append(accuracy)
        

    existing_samples = read_list()
    existing_samples.extend(samples)
    save_list(existing_samples)
    
    return existing_samples