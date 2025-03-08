import numpy as np


class LogistricRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # gradient descent
        for _ in range(self.iterations):
            # x' = wx + b
            linear_model = np.dot(X, self.weights) + self.bias
            # y = 1 / (1 + e^-x')
            y_pred = self.sigmoid(linear_model)
            dw = 1 / n_samples * np.dot(X.T, y_pred - y)
            db = 1 / n_samples * np.sum(y_pred - y)
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    regressor = LogistricRegression(learning_rate=0.0001, iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("LR classification accuracy:", accuracy(y_test, predictions))
