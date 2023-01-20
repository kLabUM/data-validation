import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

'''
Reads csv from data/ folder in as a dataframe.
    - Converts timestamp to pandas datatime datatype
'''
def import_data(index=None, node_id=None, site_id=None):
    metadata = pd.read_csv("data/metadata.csv")
    if index is not None:
        try:
            file = f"data/{metadata.iloc[index]['Node ID']}_{metadata.iloc[index]['Site ID']}.csv"
        except:
            print("Index out of bounds.")
            return
    elif node_id is not None and site_id is not None:
        file = f"data/{node_id}_{site_id}.csv"
    else:
        return
    
    try:
        df = pd.read_csv(file)
    except:
        print("Not a valid csv file.")
        return
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.index = df['Unnamed: 0']
    df = df[['Value']]
    return df

'''
Separates depth data into rising/falling sections.
    - Smooths depth values, calculates rise/run, returns original df along with falling and rising subsets
    df: dataframe with depth data (column named 'Value')
    smooth_window: window over which centered rolling mean is taken, default is 8 hours
    sensitivity: cutoff for what is considered a falling / rising data point
'''
def take_derivative(df, smooth_window='8h', sensitivity=1e-5):
    df = df.copy()
    df['smoothed'] = df['Value'].rolling(window=smooth_window, min_periods=1, center=True).mean()
    df['time diff'] = df.index.to_series().diff(periods=1).dt.seconds
    df['mean diff'] = df['smoothed'].diff(1).fillna(0) / df.index.to_series().diff(periods=1).dt.seconds 
    df = df.drop(df[df['mean diff'] < -1e100]['mean diff'].index)
    
    df_down = df[df['mean diff'] < -1*sensitivity]
    df_up = df[df['mean diff'] >= sensitivity]
    
    return df, df_down, df_up

'''
Rescales the 2D phase portrait to unit scale. 
'''
def unit_scale(df_down):
    mm = make_pipeline(MinMaxScaler())
    X = np.array(df_down[['Value', 'mean diff']])
    return mm.fit_transform(X)

'''
Removes outliers - defined as points whose change in value from previous was n_stdev away from standard deviation.
'''
def remove_outliers(df, n_stdev=3):
    df = df.copy()
    raw_diff = df['Value']
    mu = np.mean(raw_diff)
    std = np.std(raw_diff)
    thrsh_up = mu + n_stdev*std
    thrsh_down = mu - n_stdev*std
    outliers_up = df[df['Value'] - thrsh_up > 0]
    outliers_down = df[df['Value'] - thrsh_down < 0]
    outliers = pd.concat([outliers_up, outliers_down])
    
    if len(outliers)/len(df) < 0.005: # only drops outliers if they are less than .5% of the data
        df = df.drop(outliers.index)
    
    return df, outliers

'''
Clips the dataset, dropping data above the 99th percentile (for value) and below the 1st percentile (for value). 
'''
def cut_ends(df):
    q1, q99 = df['Value'].quantile([0.01, 0.99])
    a = df[df['Value']>q1]
    a = a[a['Value']<q99]
    return a
    
