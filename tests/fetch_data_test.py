from pra_ml import fetch_data
from sklearn.model_selection import train_test_split

housing = fetch_data.load_housing_data()
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

train_set, test_set = fetch_data.split_train_set(housing, 0.2)
print(len(train_set))
print(len(test_set))

housing_with_id = housing.reset_index() # add index row
print(housing_with_id)
train_set, test_set = fetch_data.split_train_test_by_id(housing_with_id, 0.2, "index")

train_set, test_set = train_test_split(housing_with_id, test_size=0.2, random_state=42)

housing.hist(bins=50, figsize=(20, 15))
#plt.show()