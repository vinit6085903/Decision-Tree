import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://github.com/vega/vega/raw/main/docs/data/seattle-weather.csv'
df = pd.read_csv(url)

# Display dataset information
print("Dataset:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display columns
print("\nColumns:")
print(df.columns)

# Define features and target variable
fe = ['temp_max', 'temp_min', 'precipitation', 'wind']
X = df[fe]
y = df['weather']

# Handle categorical target variable 'weather'
y, class_names = pd.factorize(y)  # Convert categorical labels to numerical labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
classif = DecisionTreeClassifier(random_state=42)
classif.fit(X_train, y_train)

# Make predictions and calculate accuracy
predictions = classif.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(classif, feature_names=fe, class_names=class_names, filled=True)
plt.title('Decision Tree Visualization')
plt.show()
