import pandas as pd

df = pd.read_parquet('data/neurons.parquet')
print(df.iloc[0])