import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error 
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import OrdinalEncoder  # Uncomment if you prefer ordinal

# 1. Load the data
housing = pd.read_csv("housing.csv")

# 2. Create a stratified test set based on income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Work on a copy of training data
housing = strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
# print(housing, housing_labels)

# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]
print( num_attribs, cat_attribs)

# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# housing_prepared is now a NumPy array ready for training
print(housing_prepared.shape)
print(housing_prepared)

# LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The root mean squared error for Linear Regression is: {lin_rmse}")
print("root mean squared error for Linear Regression", pd.Series(lin_rmses).describe())

# Decision Tree Regression
Dec_reg = DecisionTreeRegressor()
Dec_reg.fit(housing_prepared, housing_labels)
Dec_preds = Dec_reg.predict(housing_prepared)
# Dec_rmse = root_mean_squared_error(housing_labels, Dec_preds)
Dec_rmses = -cross_val_score(Dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The root mean squared error for Decision Tree Regression is: {Dec_rmse}")
print("root mean squared error for Decision Tree Regression", pd.Series(Dec_rmses).describe())


# RandomForestRegressor
Ran_For_reg = RandomForestRegressor()
Ran_For_reg.fit(housing_prepared, housing_labels)
Ran_For_preds = Ran_For_reg.predict(housing_prepared)
# Ran_For_rmse = root_mean_squared_error(housing_labels, Ran_For_preds)
Ran_For_rmses = -cross_val_score(Ran_For_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The root mean squared error for Random Forest Regression is: {Ran_For_rmse}")
print("root mean squared error for Random Forest Regression", pd.Series(Ran_For_rmses).describe())









"""

root mean squared error for Linear Regression count       10.000000
mean     69204.322755
std       2500.382157
min      65318.224029
25%      67124.346106
50%      69404.658178
75%      70697.800632
max      73003.752739
dtype: float64
root mean squared error for Decision Tree Regression count       10.000000
mean     69092.635536
std       2341.695652
min      65422.429145
25%      68119.226495
50%      69108.905015
75%      70199.507366
max      73503.481653
dtype: float64
root mean squared error for Random Forest Regression count       10.000000
mean     49425.787222
std       2102.650308
min      46087.894835
25%      47918.436657
50%      49274.400425
75%      50532.618765
max      53070.108837
dtype: float64
"""



"""

The root mean squared error for Linear Regression is: 69050.56219504567
root mean squared error for Linear Regression count       10.000000
mean     69204.322755
std       2500.382157
min      65318.224029
25%      67124.346106
50%      69404.658178
75%      70697.800632
max      73003.752739
dtype: float64
The root mean squared error for Decision Tree Regression is: 0.0
root mean squared error for Decision Tree Regression count       10.000000
mean     69490.919588
std       2511.675079
min      64283.876304
25%      69113.027617
50%      69733.520883
75%      70680.477085
max      72812.388372
dtype: float64
The root mean squared error for Random Forest Regression is: 18424.718790733157
root mean squared error for Random Forest Regression count       10.000000
mean     49386.516147
std       2134.243921
min      46098.602629
25%      47781.128542
50%      49196.469618
75%      50536.604832
max      53302.567085
dtype: float64
(base) PS C:\coding_1\99999\DATA-SCIENCE\scikit-learn(sec7)\Project Gurgaon> 
"""