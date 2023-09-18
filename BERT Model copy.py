import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from joblib import dump
import random
import os
import tensorflow as tf

def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # If using tensorflow
    tf.random.set_seed(seed_value)

# Set a seed value for reproducibility
print("Setting seed...")
set_seed(42)

# Define file paths to your CSV data
file_paths = ['dataset/A SYLLABUS GENERATOR FOR THE COLLEGE.csv']

# Load the data from the CSV files
print("Loading data...")
imrad_data = pd.concat([pd.read_csv(path, encoding='Windows-1252') for path in file_paths], ignore_index=True)

# Drop rows with missing labels from the data
print("Cleaning data...")
imrad_data.dropna(subset=['Label'], inplace=True)

# Convert the text data and labels to lists for processing
text_data = imrad_data['Text'].astype(str).tolist()
imrad_labels = imrad_data['Label'].tolist()

# Map the labels to numerical values
imrad_label_map = {'Introduction': 0, 'Method': 1, 'Result': 2, 'Discussion': 3}
imrad_labels = [imrad_label_map[label] for label in imrad_labels]

# Preprocess the text data by removing special characters and numbers
print("Preprocessing data...")
text_data = [re.sub(r'\W', ' ', str(x)) for x in text_data]

# Extract features from the text data using TF-IDF vectorization
print("Extracting features...")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(text_data)

# Handle class imbalance in the data using SMOTE oversampling
print("Handling class imbalance...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, imrad_labels)

# Split the resampled data into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define a parameter grid for hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth' : [None, 5, 10],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# Train a RandomForestClassifier model using GridSearchCV for hyperparameter tuning
print("Training model...")
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=cv)
grid_search.fit(X_train, y_train)

# Use the best estimator from GridSearchCV to make predictions on the test set
print("Making predictions...")
model = grid_search.best_estimator_
y_pred = model.predict(X_test)

# Group consecutive segments with the same label for post-processing analysis
print("Grouping segments...")
from itertools import groupby
grouped_segments = [(label, list(group)) for label, group in groupby(zip(X_test, y_pred), lambda x: x[1])]

# Evaluate the model's performance using a classification report and confusion matrix
print("Evaluating model...")
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Save the trained model and vectorizer for future use
print("Saving model and vectorizer...")
dump(model, 'random_forest.joblib') 
dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Done!")
