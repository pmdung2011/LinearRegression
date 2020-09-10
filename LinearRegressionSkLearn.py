import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')
X = data.iloc[:, 0:8]  # Get the first eight columns of data
Y = data.iloc[:, -1]
# Split dataset to training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# STANDARDIZE DATA
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

reg = LinearRegression().fit(X_train, Y_train)
Y_predict = reg.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y_test, Y_predict))
# r2 = reg.score(X, Y)

print(rmse)
# print(r2)