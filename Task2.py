import pandas as pd

# Load the dataset
data = pd.read_csv('train.csv')

print("Missing Values\n\n",data.isnull().sum(),"\n")

# Filling the missing values in the 'Age' column with the median age
data['Age'] = data['Age'].fillna(data['Age'].median())

# Filling the missing values in the 'Embarked' column with the most common value (mode)
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Dropping the 'Cabin' column due to high number of missing values
data = data.drop(columns=['Cabin'])

# Converting categorical columns to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Selecting only numeric columns for statistical calculations
numeric_data = data.select_dtypes(include='number')

print("Sample of cleaned data:","\n")
print(data.head(3))

# Calculating statistical measures for numeric columns
statistics = {
    'Mean': numeric_data.mean(),
    'Median': numeric_data.median(),
    'Mode': numeric_data.mode().iloc[0],  # Using iloc[0] to get the first mode for each column
    'Standard Deviation': numeric_data.std()
}

# Converting statistics dictionary to a DataFrame for easier display
statistics_df = pd.DataFrame(statistics)

# Displaying the cleaned data sample and statistics summary
print("Statistics summary:")
print(statistics_df)
