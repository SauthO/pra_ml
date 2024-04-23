from pra_ml import fetch_data as fd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#fd.fetch_housing_data()


housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

def split_train_set(df, test_ratio):
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return df.iloc[train_indices], df.iloc[test_indices] # iloc is Pandas method

train_set, test_set = split_train_set(housing, 0.2)
print(len(train_set))
print(len(test_set))
housing.hist(bins=50, figsize=(20, 15))
#plt.show()