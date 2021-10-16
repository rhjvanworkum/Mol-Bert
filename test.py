import pandas as pd

test = pd.read_csv('C:/users/rhjva/downloads/chembl_29_chemreps.txt', sep='\t')

print(len(test.index))
print(test.head())