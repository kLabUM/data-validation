from utils import *
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

class ReportErrors():
    
    
#     def create_phase_portrait(self):
        
#         df = self.df.copy()
#         df, df_down, df_up = take_derivative(df, smooth_window='6h')

    '''
    Calculates optimal eps based on KNN method described by !!!!!!!<paper name>!!!!!!!!
    '''
    def return_eps(X, quantile=0.9975):
        if len(X) < 2:
            return 0.025
        else:
            nbrs = NearestNeighbors(n_neighbors=2).fit(X) # looks at the difference between all pairs of pts
            distances, indices = nbrs.kneighbors(X)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            q995 = np.quantile(distances,quantile) # we want close to the max difference between points in this set
            return q995
    
    '''
    Applies DBSCAN clustering to the phase portrait, returning labels. 
    '''
    def DBSCAN_label(X, eps=0.015, min_samples=50):
        labels = np.zeros(len(X))
        if len(X) > 0:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = db.labels_
        return labels
    
    '''
    Detects data that are outside the physical range of the sensor.
    - neg_bound: the minimum physically feasible value (default: -10 cm)
    - sat_bound: the minimum distance from sensor before saturation occurs (default: 500 mm below offset)
    '''
    def detect_out_of_range(df, offset, neg_bound=-100, sat_bound=500):
    
        neg_flag = False
        sat_flag = False
        corrected = False
        df, df_down, df_up = take_derivative(df)

        # detect negative values (out of range)
        negative = df[df['Value'] < neg_bound]
        if len(negative) < 0.25*len(df):
            df = df.drop(negative.index)
            if len(negative) > 0.01*len(df): # if between 1% and 25% are negative
                neg_flag = True
        elif len(negative) > 0.25*len(df): # if greater than 25% are negative, we assume the offset is wrong
            df['Value'] = df['Value'] + 1000
            corrected = True

        # detect saturation (out of range)
        if corrected:
            sat_threshold = 1000 + offset - sat_bound
        else:
            sat_threshold = offset - sat_bound
            
        too_close = df[df['Value'] >= sat_threshold]
            
        if len(too_close) > 0.005*len(df): # if greater than .5% of the data are saturated
            sat_flag = True
        df = df.drop(too_close.index)

        df, df_down, df_up = take_derivative(df) # recalculate derivative

        return df, neg_flag, sat_flag
    
    '''
    Detects underlying outliers (greater than n interquartile ranges below mean)
    - n: number of interquartile ranges below mean we set the threshold (default: 3)
    '''
    def detect_underliers(df, n=3):
    
        flag = False

        total_underliers = 0
        breakpoints = separate_seasons(df) #separate seasons if applicable
        for i in range(len(breakpoints) -1): # for every subset of data
            subset = df[breakpoints[i]:breakpoints[i+1]]
            q1, q2, q3 = subset['Value'].quantile([0.25, 0.5, 0.75])
            min_bed = q2 - n*(q3-q1)
            underliers = subset[subset['Value']< min_bed]
            total_underliers += len(underliers)

        if total_underliers > 0.01*len(df) and total_underliers < 0.1*len(df): # flag if 1-10% of data shows error
            flag = True
            df = df.drop(underliers.index)
            
        df, df_down, df_up = take_derivative(df) # recalculate derivative

        return df, flag
    
    def threshold_solid_obstruction(X, n_bins=6):
        h, xedges, yedges = np.histogram2d(X[:,0],X[:,1],n_bins)
        v = np.reshape(h,(n_bins**2,))
        a = v/sum(v)
        norm_h = np.reshape(a,(n_bins,n_bins))
        row_sums = np.sum(norm_h, axis=0)
        col_sums = np.sum(norm_h, axis=1)
        min_row = np.argmin(row_sums)
        min_col = np.argmin(col_sums)

        thrsh = {}
        obstruction = False
        indeterminate = False
        if min_row < 5 and min_row > 0:
            if row_sums[min_row] < 0.1:
                for j in range(min_row+1,6):
                    if row_sums[j] > 2*row_sums[min_row]:
                        obstruction = True
                        thrsh['y'] = yedges[min_row+1]
        if min_col==3 or min_col==4:
            for j in range(min_col+1,6):
                if col_sums[j] > col_sums[min_col]:
                    if col_sums[j] > 0.05:
                        obstruction = True
                        thrsh['x'] = -1*xedges[min_col]

        return thrsh

    def detect_solid_obstruction(thrsh, df_down):
        obstruction = False
        indeterminate = False
        if 'y' in thrsh.keys():
            threshold = (df_down['Value'].max() - df_down['Value'].min())*thrsh['y'] + df_down['Value'].min()
            obst = df_down[df_down['Value'] > threshold]

            if len(obst) > len(df_down)*0.01: # could be an obstruction if its > 1% of points
                obstruction = True
            else:
                coeff_var = np.sqrt(np.var(obst['Value']))/np.mean(obst['Value'])
                if coeff_var < 0.05:
                    indeterminate = True
        else:
            threshold = df_down['Value'].max()
        return obstruction, indeterminate, threshold
    
            
            
        
        
        