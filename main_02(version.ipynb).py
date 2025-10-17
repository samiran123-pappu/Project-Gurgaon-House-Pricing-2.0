import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("housing.csv")

# Create income category bins and labels
bins = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, np.inf]  # bin edges
labels = [1, 2, 3, 4, 5, 6, 8]  # category labels
df["income_cat"] = pd.cut(df["median_income"], bins=bins, labels=labels)  # new column

# Plot the distribution of income categories
df["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)  # bar plot
plt.title("Income Categories Distribution")
plt.xlabel("Income Category")
plt.ylabel("Number of Instances")
plt.show()

# Display dataset overview
df.tail()       # last few rows
df.info()       # info
df.describe()   # stats
df["median_house_value"].value_counts()  # count values

# Create histograms for numerical columns
from pandas.plotting import scatter_matrix
attributes = ["housing_median_age","median_income", "median_house_value" ]
scatter_matrix(df[attributes],figsize = (12, 8) )


# Perform stratified train-test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # split
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]  # train set
    strat_test_set = df.loc[test_index]    # test set

# Remove income_cat column from both sets
for sett in (strat_test_set, strat_train_set):
    sett.drop("income_cat", axis=1, inplace=True)  # drop column

# Create working copy of training set
df = strat_train_set.copy()

# Separate features and labels
housing = strat_train_set.drop("median_house_value", axis=1)  # features
housing_labels = strat_train_set["median_house_value"].copy()  # labels

# Handle missing values with imputer
imputer = SimpleImputer(strategy="median")  # imputer
housing_num = housing.select_dtypes(include=[np.number])  # numeric features
imputer.fit(housing_num)  # fit
X = imputer.transform(housing_num)  # transform
X = imputer.transform(housing_num)                      #Transform Array to Data_Frame
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_tr["ocean_proximity"] = df["ocean_proximity"] #Adding ocean_proximity

import pandas as pd    #handle ocean_proximity by OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Assuming your DataFrame is called 'housing_data'
# Extract just the ocean_proximity column
ocean_proximity_col = housing_tr[['ocean_proximity']]

# One-hot encode only the ocean_proximity column
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(ocean_proximity_col)

# Convert to array (if needed)
housing_cat_1hot_array = housing_cat_1hot.toarray()

# Create DataFrame with proper column names
feature_names = cat_encoder.get_feature_names_out(['ocean_proximity'])
housing_catt_4 = pd.DataFrame(housing_cat_1hot_array,columns=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
,index=housing_tr.index)

print(housing_catt_4.head())

df = pd.concat([housing_tr,housing_catt_4 ], axis = 1)
print(df)




