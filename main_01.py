import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Load the original dataset
original_data = pd.read_csv("housing.csv")

# 2. Create a stratified test set based on income category
original_data["income_cat"] = pd.cut(
    original_data["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(original_data, original_data["income_cat"]):
    stratified_train_data = original_data.loc[train_index].drop("income_cat", axis=1)
    stratified_test_data = original_data.loc[test_index].drop("income_cat", axis=1)

# 3. Work on a copy of training data
train_data = stratified_train_data.copy()

# 4. Separate predictors and labels
train_labels = train_data["median_house_value"].copy()
train_features = train_data.drop("median_house_value", axis=1)
print(train_features, train_labels)

# 5. Explicitly set numerical and categorical columns
numerical_columns = train_features.drop("ocean_proximity", axis=1).columns.tolist()
categorical_columns = ["ocean_proximity"]

print("\nNumerical attributes:", numerical_columns)
print("Categorical attributes:", categorical_columns)

# 6. Pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine pipelines
full_pipeline = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns),
])
# 8. Transform the data
prepared_features = full_pipeline.fit_transform(train_features)
print(prepared_features.shape)
print(prepared_features)






# 9. Convert the transformed features into a DataFrame with proper column names
num_feature_names = numerical_columns

# Get one-hot encoded categorical feature names
cat_feature_names = list(
    full_pipeline.named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_columns)
)

# Combine them
all_feature_names = num_feature_names + cat_feature_names

# Convert prepared_features into a DataFrame
prepared_df = pd.DataFrame(
    prepared_features.toarray() if hasattr(prepared_features, "toarray") else prepared_features,
    columns=all_feature_names,
    index=train_features.index
)

print(prepared_df)
print("\nShape of prepared DataFrame:", prepared_df.shape)




# Output
"""
(base) PS C:\coding_1\99999\DATA-SCIENCE\scikit-learn(sec7)\Project Gurgaon> & C:/Users/samir/anaconda3/python.exe "c:/coding_1/99999/DATA-SCIENCE/scikit-learn(sec7)/Project Gurgaon/main_01.py"
       longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income ocean_proximity
12655    -121.46     38.52                  29         3873           797.0        2237         706         2.1736          INLAND
15502    -117.23     33.09                   7         5320           855.0        2015         768         6.3373      NEAR OCEAN
2908     -119.04     35.37                  44         1618           310.0         667         300         2.8750          INLAND
14053    -117.13     32.75                  24         1877           519.0         898         483         2.2264      NEAR OCEAN
20496    -118.70     34.28                  27         3536           646.0        1837         580         4.4964       <1H OCEAN
...          ...       ...                 ...          ...             ...         ...         ...            ...             ...
15174    -117.07     33.03                  14         6665          1231.0        2026        1001         5.0900       <1H OCEAN
12661    -121.42     38.51                  15         7901          1422.0        4769        1418         2.8139          INLAND
19263    -122.72     38.44                  48          707           166.0         458         172         3.1797       <1H OCEAN
19140    -122.70     38.31                  14         3155           580.0        1208         501         4.1964       <1H OCEAN
19773    -122.14     39.97                  27         1079           222.0         625         197         3.1319          INLAND

[16512 rows x 9 columns] 12655     72100
15502    279600
2908      82700
14053    112500
20496    238300
          ...
15174    268500
12661     90400
19263    140400
19140    258100
19773     62700
Name: median_house_value, Length: 16512, dtype: int64

Numerical attributes: ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
Categorical attributes: ['ocean_proximity']
(16512, 13)
[[-0.94135046  1.34743822  0.02756357 ...  0.          0.
   0.        ]
 [ 1.17178212 -1.19243966 -1.72201763 ...  0.          0.
   1.        ]
 [ 0.26758118 -0.1259716   1.22045984 ...  0.          0.
   0.        ]
 ...
 [-1.5707942   1.31001828  1.53856552 ...  0.          0.
   0.        ]
 [-1.56080303  1.2492109  -1.1653327  ...  0.          0.
   0.        ]
 [-1.28105026  2.02567448 -0.13148926 ...  0.          0.
   0.        ]]
       longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  ocean_proximity_<1H OCEAN  ocean_proximity_INLAND  ocean_proximity_ISLAND  ocean_proximity_NEAR BAY  ocean_proximity_NEAR OCEAN
12655  -0.941350  1.347438            0.027564     0.584777        0.640371    0.732602    0.556286      -0.893647                        0.0                     1.0                     0.0                       0.0                         0.0
15502   1.171782 -1.192440           -1.722018     1.261467        0.781561    0.533612    0.721318       1.292168                        0.0                     0.0                     0.0                       0.0                         1.0
2908    0.267581 -0.125972            1.220460    -0.469773       -0.545138   -0.674675   -0.524407      -0.525434                        0.0                     1.0                     0.0                       0.0                         0.0
14053   1.221738 -1.351474           -0.370069    -0.348652       -0.036367   -0.467617   -0.037297      -0.865929                        0.0                     0.0                     0.0                       0.0                         1.0
20496   0.437431 -0.635818           -0.131489     0.427179        0.272790    0.374060    0.220898       0.325752                        1.0                     0.0                     0.0                       0.0                         0.0
...          ...       ...                 ...          ...             ...         ...         ...            ...                        ...                     ...                     ...                       ...                         ...
15174   1.251711 -1.220505           -1.165333     1.890456        1.696862    0.543471    1.341519       0.637374                        1.0                     0.0                     0.0                       0.0                         0.0
12661  -0.921368  1.342761           -1.085806     2.468471        2.161816    3.002174    2.451492      -0.557509                        0.0                     1.0                     0.0                       0.0                         0.0
19263  -1.570794  1.310018            1.538566    -0.895802       -0.895679   -0.862013   -0.865118      -0.365475                        1.0                     0.0                     0.0                       0.0                         0.0
19140  -1.560803  1.249211           -1.165333     0.249005        0.112126   -0.189747    0.010616       0.168261                        1.0                     0.0                     0.0                       0.0                         0.0
19773  -1.281050  2.025674           -0.131489    -0.721836       -0.759358   -0.712322   -0.798573      -0.390569                        0.0                     1.0                     0.0                       0.0                         0.0

"""