import pandas as pd
from apyori import apriori

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/Groceries_dataset.csv')

# Preprocess the data
basket = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index(name='items')
#.apply(list)  used to change datatype to list 
#reset_index(name='items')   resets the index of the grouped data and assigns a new name 'items' to the column containing the lists of items
print(basket)
# Convert the transaction data to a list of lists
transactions = basket['items'].tolist()
#print(transactions)
# Perform Apriori association rule mining
association_rules = list(apriori(transactions))

# Print the results
for rule in association_rules:
    print(rule)
