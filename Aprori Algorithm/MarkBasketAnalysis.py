import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori

df = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv')

data = df.copy()
data1 = data.copy()

data=pd.get_dummies(df['itemDescription']).astype(int)
print(data)

data1.drop(['itemDescription'], axis=1, inplace=True)

data1 = data1.join(data)

products = df['itemDescription'].unique()

data2 = data1.groupby(['Member_number', 'Date'])[products[:]].sum()

data2 = data2.reset_index()[products]

def func(data):
  for i in products:
    if data[i] > 0:
      data[i] = i
  return data

data2 = data2.apply(func,axis=1)
newdata = data2.values
newdata = [i[i!=0].tolist() for i in newdata if i[i!=0].tolist()]

association = apriori(newdata, 
                      min_support=0.0003,
                      min_confidence=0.05,
                      min_lift=3,
                      max_length=2)

result = list(association)

for i in result[5]:
  print(i)