# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Wine Quality Red dataset
df = pd.read_csv('Data Preprocessing/winequality.csv', delimiter=';')
# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(y)
# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the StandardScaler class
sc = StandardScaler()

# Transform the training and test sets using the fitted scaler
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Scaled training set:\n", X_train)
print("Scaled test set:\n", X_test)