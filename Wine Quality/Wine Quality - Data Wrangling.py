import pandas as pd

### https://www.kaggle.com/code/kardelenolakolu/swe584-term-project
### https://www.kaggle.com/code/lizk75/red-wine-svm-90-accuracy
#df = pd.read_csv("D:\\PyCharm\\BroCode\\Data\\WineQT.csv")
df= pd.read_csv("Data/WineQT.csv")
print(df.head())

pd.set_option('display.max_columns', None)

df=df.drop(['Id'], axis=1)
df['quality'] = df['quality']-3
print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df['quality'].nunique())
print(df['quality'].unique())
print(df['quality'].value_counts()/len(df))

missing_val = df.isin(['?', '', 'None', 'NaN']).sum()
print(missing_val)

#df.describe()[1:].T.style.background_gradient(cmap='Blues')
print(df.describe()[1:].T)