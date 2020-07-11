import numpy as np

def take_log(df):
    # take log of data
    df['Adj Close'] = np.log(df['Adj Close'])
    
    return df

def get_deltas(df):
    vals = df['Adj Close']
    deltas = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    deltas.append(np.nan)
    df['Adj Close'] = deltas
    df.to_csv('./data/processed/deltas.csv')
    #df.to_csv('../data/processed/deltas.csv')
    return df, deltas


def scale_data(df):
    scaler = MinMaxScaler(feature_range = (0, 1))
    df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])
    # save scaler to use later when interpretting predictions
    joblib.dump(scaler, './models/MinMaxScaler.save')
    #joblib.dump(scaler, '../models/MinMaxScaler.save')
    return df


def create_previous_days(df, col_name):
    
    """
    Create columns containing the stock price for each of the previous five days.
    
    Params
    ------
    DataFrame containing stock prices for consecutive days
    
    Return
    ------
    A times series formatted DataFrame
    """
    
    back_1 = np.nan
    back_2 = np.nan
    back_3 = np.nan
    back_4 = np.nan
    back_5 = np.nan
    
    back_1_col = []
    back_2_col = []
    back_3_col = []
    back_4_col = []
    back_5_col = []
    
    for today in df[col_name]:
        
        # append previous values
        back_1_col.append(back_1)
        back_2_col.append(back_2)
        back_3_col.append(back_3)
        back_4_col.append(back_4)
        back_5_col.append(back_5)
        
        # set values for next day to step forward
        back_5 = back_4
        back_4 = back_3
        back_3 = back_2
        back_2 = back_1
        back_1 = today
        
    # append columns to time_series dataframe
    df['back_5'] = back_5_col
    df['back_4'] = back_4_col
    df['back_3'] = back_3_col
    df['back_2'] = back_2_col
    df['back_1'] = back_1_col

    # order columns chronologically
    df = df[['back_5', 'back_4', 'back_3', 'back_2', 'back_1', col_name]]
        
    return df



    df = df.sort_index(ascending=False)
    
    next_1 = np.nan
    next_2 = np.nan
    next_3 = np.nan
    next_4 = np.nan
    next_5 = np.nan
    
    next_1_col = []
    next_2_col = []
    next_3_col = []
    next_4_col = []
    next_5_col = []
    
    for today in df[col_name]:
        
        next_1_col.append(next_1)
        next_2_col.append(next_2)
        next_3_col.append(next_3)
        next_4_col.append(next_4)
        next_5_col.append(next_5)
        
        next_5 = next_4
        next_4 = next_3
        next_3 = next_2
        next_2 = next_1
        next_1 = today
        
    df['next_1'] = next_1_col
    df['next_2'] = next_2_col
    df['next_3'] = next_3_col
    df['next_4'] = next_4_col
    df['next_5'] = next_5_col
    
    df = df.sort_index(ascending=True)
    
    return df

def create_future_days(df, col_name):
    
    df = df.sort_index(ascending=False)
    
    next_1 = np.nan
    next_2 = np.nan
    next_3 = np.nan
    next_4 = np.nan
    next_5 = np.nan
    
    next_1_col = []
    next_2_col = []
    next_3_col = []
    next_4_col = []
    next_5_col = []
    
    for today in df[col_name]:
        
        next_1_col.append(next_1)
        next_2_col.append(next_2)
        next_3_col.append(next_3)
        next_4_col.append(next_4)
        next_5_col.append(next_5)
        
        next_5 = next_4
        next_4 = next_3
        next_3 = next_2
        next_2 = next_1
        next_1 = today
        
    df['next_1'] = next_1_col
    df['next_2'] = next_2_col
    df['next_3'] = next_3_col
    df['next_4'] = next_4_col
    df['next_5'] = next_5_col
    
    df = df.sort_index(ascending=True)
    
    return df