# Module to load and preprocess data
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


# Function to load dataset given its UCI repository ID (mushroom_id=73, wine_id=187)
def load(dataset_id, target_column):
    dataset = fetch_ucirepo(id=dataset_id)

    # Features and target are already pandas objects
    X = dataset.data.features
    y = dataset.data.targets

    # Ensure target is a DataFrame
    if not isinstance(y, pd.DataFrame):
        y = y.to_frame(name=target_column)

    # Combine into one DataFrame (convenient for preprocessing)
    data = pd.concat([X, y], axis=1)

    return data
  
# Function to preprocess data: handle missing values, encode categorical variables, normalise numerical features
def one_hot_encode(data, exclude_columns=[]):

    # Exclude specified columns from one-hot encoding
    data = data.copy()
    cols_to_encode = [col for col in data.columns if col not in exclude_columns]

    # One-hot encode categorical features
    encoded_data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)  # drop_first to avoid dummy variable trap
    return encoded_data

# Function to normalise numerical features
def normalise_features(data, exclude_columns=[]):

    # Normalise numerical features to have mean 0 and std 1
    numerical_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_columns]
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()
    return data

# Function to handle missing values
def handle_missing_values(data):

    # Replace '?' with NaN explicitly (mushroom dataset)
    data = data.replace("?", np.nan)

    for col in data.columns:
        if data[col].dtype == np.number:
            data[col].fillna(data[col].mean(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

    return data


# Manual train-test split
def train_test_split(X, y, test_ratio=0.2, seed=42):
    np.random.seed(seed) # for reproducibility
    idx = np.random.permutation(len(X)) # shuffle indices
    t = int(len(X) * test_ratio) # test set size
    return X[idx[t:]], X[idx[:t]], y[idx[t:]], y[idx[:t]] # return train and test splits
