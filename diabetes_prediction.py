import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Load and Explore the Dataset
# Load the dataset
df = pd.read_csv('diabetes.csv')

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Show basic statistics
print("\nDataset statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

## Visualize the Data
# Plot the distribution of Glucose levels
plt.hist(df['Glucose'], bins=20, color='skyblue')
plt.title('Glucose Level Distribution')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

## Prepare the Data for Machine Learning
# Separate features (X) and the target variable (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Train a Random Forest Model
# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

