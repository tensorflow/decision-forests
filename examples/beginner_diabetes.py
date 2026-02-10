# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This example trains, display and evaluate a Random Forest model on the pima India's Diabetes dataset

This example works with the pip package.

Usage example (in a shell):

  pip3 install tensorflow_decision_forests
  python3 beginner_diabetes.py

More examples are available in the documentation's colabs.
"""

"""About

TensorFlow Decision Forests (TF-DF) is a collection of state-of-the-art algorithms for the training,
 serving and interpretation of Decision Forest models. The library is a collection of Keras models 
 and supports classification, regression and ranking.
 for more details [link](https://pypi.org/project/tensorflow-decision-forests/)
"""

# NOTE: Uncomment the command below if you don't have tensorflow_decision_forests package installed
# !pip install tensorflow_decision_forests

# Python libraries
# Classic,data manipulation and linear algebra
import pandas as pd
import numpy as np

# Data processing, metrics and modeling
import tensorflow_decision_forests as tfdf

# Check the current version of TensorFlow Decision Forests
print("Found TF-DF v" + tfdf.__version__)

"""
Dataset used in this notebook can be found [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
"""

# loading dataset
pima = pd.read_csv(".datasets/diabetes.csv")

pima.head()

#selecting the important features and target variable
feature_cols = ['Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction', 'Outcome']
dataset_df = pima[feature_cols]

# Split the dataset into a training and a testing dataset into 70-30 ratio.
test_indices = np.random.rand(len(dataset_df)) < 0.30
test_ds_pd = dataset_df[test_indices]
train_ds_pd = dataset_df[~test_indices]
print(f"{len(train_ds_pd)} examples in training"
      f", {len(test_ds_pd)} examples for testing.")

# Converts a Pandas dataset into a tensorflow dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label="Outcome")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label="Outcome")

# Trains the model.
model = tfdf.keras.RandomForestModel(verbose=2)
model.fit(x=train_ds)

# Summary of the model structure.
model.summary()

# Evaluate the model on the validation dataset.
model.compile(metrics=["accuracy"])
evaluation = model.evaluate(test_ds)

# Export the model to the SavedModel format for later re-use e.g. TensorFlow
# Serving.
model.save("/temp/my_saved_model")

# Look at the feature importances.
model.make_inspector().variable_importances()

