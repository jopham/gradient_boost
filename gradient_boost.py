'''

This project explores gradient boosting using CatBoost (based on a tutorial).
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
from sklearn.model_selection import train_test_split


#######################
#   DATA EXPLORATION
#######################

#Set the seed
random_seed = 12345

# Read in the dataset
from catboost.datasets import amazon
(train_df, test_df) = amazon()

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
features = dict(list(enumerate(train_df.keys()[1:])))

create_cd(
    label=0,
    cat_features=list(range(1, train_df.shape[1])),
    feature_names=features,
    output_path=os.path.join(dataset_dir, 'train.cd')
)

# Store data set in a Pool class
# *Notes: Tinker with Pool more
pool_b = Pool(
    data=os.path.join(dataset_dir, 'train.csv'),
    delimiter=',',
    column_description=os.path.join(dataset_dir, 'train.cd'),
    has_header=True
)

# Check the contents
print('Dataset shape: {}\n'.format(pool_b.shape))
print('Column names: {}'.format(pool_b.get_feature_names()))

# Fit models
CatBoostClassifier(iterations=3).fit(X, y, cat_features=cat_features);
CatBoostClassifier(iterations=3).fit(pool_a)
CatBoostClassifier(iterations=3).fit(pool_b)

# Split into train & validation sets
data = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train, X_validation, y_train, y_validation = data

train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features
)

validation_pool = Pool(
    data=X_validation,
    label=y_validation,
    cat_features=cat_features
)


#######################
#   BETTER/BEST MODEL
#######################

# Note: You can tinker with learning rates
model = CatBoostClassifier(
    iterations=5,
    learning_rate=0.1,
)
model.fit(train_pool, eval_set=validation_pool, verbose=False)

# Print model info
print('Model is fitted: {}'.format(model.is_fitted()))
print('Model params:\n{}'.format(model.get_params()))

# Choose the best iteration
# Note: There is a parameter: use_best_model ( = True or False)
model = CatBoostClassifier(
    iterations=100,
)

model.fit(
    train_pool,
    eval_set=validation_pool,
    verbose=False,
)

print('Tree count: ' + str(model.tree_count_))

