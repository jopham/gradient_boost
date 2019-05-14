'''

This project explores gradient boosting using CatBoost.
Data: Amazon Employee Access Challenge

'''

#######################
#   IMPORT & DATA
#######################

#!pip install --user --upgrade catboost
import os
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)

import catboost
from catboost import CatBoostClassifier
from catboost import Pool
from catboost.utils import create_cd

# Read in the dataset
from catboost.datasets import amazon


#######################
#   DATA EXPLORATION
#######################

# Extract labels
y = train_df.ACTION
X = train_df.drop('ACTION', axis=1)

# Define categorical variables
cat_features = list(range(0, X.shape[1]))
print(cat_features)


#######################
#   BASIC MODEL
#######################

# Create a basic model with 200 trees
model = CatBoostClassifier(iterations=200)
model.fit(X, y, cat_features=cat_features, verbose=5)
# Create predictions for X
model.predict_proba(X)

# Store the data set in a Pool class for training
pool_a = Pool(data=X, label=y, cat_features=cat_features)

# Define the directory, create a new one it if does not exist
dataset_dir = './amazon'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Create train.csv
train_df.to_csv(
    os.path.join(dataset_dir, 'train.csv'),
    index=False, sep=',', header=True
)

# Create column descriptions


