import pandas as pd
from sklearn import model_selection

df = pd.read_csv("data/full.csv")
train, test = model_selection.train_test_split(df, test_size=0.1, random_state=2)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)