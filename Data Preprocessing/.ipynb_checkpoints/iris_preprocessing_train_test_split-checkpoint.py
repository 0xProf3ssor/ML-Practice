# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Iris dataset
df = pd.read_csv('/media/Projects/python/ML-Practice/Data Preprocessing/iris.csv')

# Separate features and target
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Apply feature scaling on the training and test sets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Display the first few rows of the scaled training set
print(X_train[:5])
# Print the scaled training and test sets
print('Training set:', X_train)
print('Test set:', X_test)