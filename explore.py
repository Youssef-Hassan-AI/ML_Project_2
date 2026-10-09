import pandas as pd

df = pd.read_csv('my_dataset.csv', nrows=5000)
print('Shape:', df.shape)
print('\nDtypes:')
print(df.dtypes)
print('\nNull counts:')
print(df.isnull().sum())
print('\nTarget (fitness_level) stats:')
print(df['fitness_level'].describe())
print('\nUnique values per col:')
for c in df.columns:
    print(f'  {c}: {df[c].nunique()}')
print('\nCategorical cols sample:')
for c in df.select_dtypes(include='object').columns:
    print(f'  {c}: {df[c].unique()[:10]}')
print('\nfitness_level value range:')
print(f'  min={df["fitness_level"].min()}, max={df["fitness_level"].max()}')
print(f'  sample values: {sorted(df["fitness_level"].unique())[:20]}')
