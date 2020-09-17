import pandas as pd

def read_ex2data1():
    df = pd.read_csv("ex2data1.csv")
    X = df[['x1', 'x2']].values.reshape(-1, 2)
    Y = df['y'].values.reshape(-1)
    return X, Y