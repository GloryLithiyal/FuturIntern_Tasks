import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('train.csv')
print(data.head(5), "\n")

#Step1: Missing Values
print("Missing Values\n\n",data.isnull().sum())
"\n"
data.describe()

# Step 2: Remove Missing Values
data_no_missing = data.dropna()
print("\nData After Removing Missing Values:\n", data_no_missing.isnull().sum())

# Step 3: Check for Duplicates
print("\nNumber of duplicate rows:", data_no_missing.duplicated().sum())

# Step 4: Outlier Detection for Age and Fare using Z-score Method
columns = ["Age", "Fare"]

# Calculate Z-scores for Class and Fare
z_scores = np.abs((data_no_missing[columns] - data_no_missing[columns].mean()) / data_no_missing[columns].std())

# Identify and display the rows where Z-scores are greater than 3 for Class or Fare
outliers = data_no_missing[(z_scores > 3).any(axis=1)][columns]
print("\nOutliers in Age and Fare using Z-score:\n", outliers)

# Step 5: Remove Outliers from the Data
data_no_outliers = data_no_missing[(z_scores <= 3).all(axis=1)][columns]

# Display cleaned data without outliers
print("\nData after removing outliers:\n", data_no_outliers)

data_no_outliers.to_csv('cleaned_dataset.csv', index=False)
