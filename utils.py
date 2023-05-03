import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

'''
Reads csv from data/ folder in as a dataframe.
    - Converts timestamp to pandas datatime datatype
'''
def import_wl_data(index=None, node_id=None, site_id=None):
    metadata = pd.read_csv("data/metadata.csv")
    if index is not None:
        try:
            file = f"data/water_level/{metadata.iloc[index]['Node ID']}_{metadata.iloc[index]['Site ID']}.csv"
        except:
            print("Index out of bounds.")
            return pd.DataFrame({})
    elif node_id is not None and site_id is not None:
        file = f"data/water_level/{node_id}_{site_id}.csv"
    else:
        return pd.DataFrame({})
    
    try:
        df = pd.read_csv(file)
    except:
        print("Not a valid csv file.")
        return
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.index = df['Unnamed: 0']
    df = df[['Value']]
    return df

def import_ph_data(node_id, start, end):
    df = pd.read_csv(f"data/pH_sensors/{node_id}.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.index = df['Unnamed: 0']
    df = df[['Value']]
    df = df[start:end]
    return df

def import_ec_data(node_id, start, end):
    df = pd.read_csv(f"data/ec_sensors/{node_id}.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.index = df['Unnamed: 0']
    df = df[['Value']]
    df = df[start:end]
    return df

def grab_data(node_id, measurement, start, end):
    tags = {'node_id':f"{node_id}"}
    data = Query.run_query(field='value', measurement=measurement, tags=tags, df=True)
    data = data[start:end]
    return data

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
# def cut_ends(df):
#     try:
# #     if len(df) > 100:
#         q1, q99 = df['Value'].quantile([0.01, 0.99])
#         a = df[df['Value']>q1]
#         a = a[a['Value']<q99]
#         return a
#     except:
#         return df
    
def cut_ends(df):
    try:
        q1, q99 = df['Value'].quantile([0.005, 0.995])
        a = df.copy()
        if len(df[df['Value']<q1]) < 0.02*len(df): 
            a = df[df['Value']>q1]
        if len(a[a['Value']>q99]) < 0.02*len(df):
            a = a[a['Value']<q99]
        return a
    except:
        return df
    
'''
Looks for intersections in a 3 month rolling mean and a year-long rolling mean to identify seasonality in the data. 
''' 
def separate_seasons(df):
    df = df.copy()
    if len(df) > 0:
        breakpoints = [df.index[0]]
        df['season'] = df['Value'].rolling('90d', min_periods=1).mean()
        df['year'] = df['Value'].rolling('365d', min_periods=1).mean()

        use = df[abs(df['year'] - df['season']) != 0]
        cuts = use[abs(use['year'] - use['season']) < 0.01].index
        for cut in cuts:
            breakpoints.append(cut)

        breakpoints.append(df.index[-1])
    else:
        breakpoints = []
    return breakpoints
