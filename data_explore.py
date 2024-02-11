import pandas as pd

data = pd.read_csv("data/full.csv").to_numpy()

veg = data[:,1:8]
bare = data[:,9:]

print(veg[0])
print(bare[0])