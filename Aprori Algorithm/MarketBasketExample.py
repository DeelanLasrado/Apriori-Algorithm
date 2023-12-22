import pandas as pd
import numpy as np
from apyori import apriori


# Create a sample DataFrame for Market Basket Analysis
data = {'Transaction': [
    ['apple', 'banana', 'orange'],
    ['banana', 'grape'],  
    ['apple', 'orange'],
    ['apple', 'banana', 'grape'],
    ['orange', 'grape']
]}

df = pd.DataFrame(data)
print(df)
# Convert the DataFrame to a list of lists
transactions_list = df['Transaction'].tolist()
'''#converting dataframe into list of lists
l=[]
for i in range(1,7501):
    l.append([str(st_df.values[i,j]) for j in range(0,20)])'''
 
# Print the resulting list
for transaction in transactions_list:
    print(transaction)

#applying apriori algorithm
association_rules = apriori(transactions_list)
association_results = list(association_rules)
print(association_rules)
print(association_results)

for i in range(0, len(association_results)):
    print(association_results[i][0])