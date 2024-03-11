import pandas as pd

labels = pd.read_csv('TEST/testing_labels')
a = labels.iloc[0, 0]
print(a)