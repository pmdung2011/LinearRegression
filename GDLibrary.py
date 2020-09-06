import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')

# X_in = np.array(data.iloc[:, 0])
# X_in = (X_in - np.mean(X_in)) / np.std(X_in)  # Normalize data
X_in = np.array((data.iloc[:, 0] - np.mean(data.iloc[:, 0])) / np.std(data.iloc[:, 0]))
X_in = X_in.reshape(-1, 1)
# X_in = np.c_[np.ones(X_in.shape[0]), X_in]
Y_in = np.array(data.iloc[:, 8])
Y_in = Y_in.reshape(-1, 1)

# Split dataset to training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X_in, Y_in, test_size=0.2, random_state=0)

# Training the algorithm
reg = LinearRegression().fit(X_train, y_train)

# Retrieve the slope (m):
print(reg.coef_)

# Retrieve the intercept (b):
print(reg.intercept_)

y_hypothesis = reg.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_hypothesis.flatten()})
print(df)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_hypothesis))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_hypothesis))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_hypothesis)))
