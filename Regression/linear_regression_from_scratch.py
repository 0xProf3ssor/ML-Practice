import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.array([])
        self.bias = np.array([])

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.gradient_descent(n_samples, n_features, X, y)

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (numerator / denominator)

    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def root_mean_squared_error(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def gradient_descent(self, n_samples, n_features, X, y):
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


# Example usage:
if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    predictions = model.predict([[6], [7]])
    print(predictions)
    predictions = model.predict(X)
    print("MSE: ", model.mean_squared_error(y, predictions))
    print("R2: ", model.r2_score(y, predictions))
    print("MAE: ", model.mean_absolute_error(y, predictions))
    print("RMSE: ", model.root_mean_squared_error(y, predictions))
