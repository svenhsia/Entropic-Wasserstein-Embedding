import numpy as np
import pandas as pd

def load_time_series(filename):
    series = pd.read_csv(filename)
    columns = series.columns.values
    raw_series = series[[col for col in columns if 'W' in col]]
    normalized_series = series[[col for col in columns if 'Normalized' in col]]
    return raw_series, normalized_series

def DTWdistance(s, t):
    n = len(s)
    m = len(t)
    DTW = np.full(shape=(n+1, m+1), fill_value=np.inf)
    DTW[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.abs(s[i-1] - t[j-1])
            DTW[i, j] = cost + min([DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1]])
    
    print(DTW)
    return DTW[n, m]

# # test
_, series = load_time_series('./data/Sales_Transactions_Dataset_Weekly.csv')
print(len(series.columns.values))

series_array = series.to_numpy()
print(series_array.shape)
n = series_array.shape[0]
distances = np.zeros((n, n))
for i in range(n):
    if (i % 100 == 0):
        print(i)
    for j in range(i, n):
        distances[i, j] = DTWdistance(series_array[i], series_array[j])
        distances[j, i] = distances[i, j]

# print(np.diag(distances))
np.savetxt('./data/Sales_Transaction_Dataset.dist', distances, delimiter=',')
