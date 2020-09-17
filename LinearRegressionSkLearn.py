import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# DEFINE DATASET
data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')
X = data.iloc[:, 0:8]  # Get the first eight columns of data
Y = data.iloc[:, -1]

# NORMALIZE DATA
X = np.array(preprocessing.normalize(X, norm='l2'))

# Split dataset to training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# DEFINE THE MODEL
model = LinearRegression()

# FIT THE MODEL
reg = model.fit(X_train, Y_train)
Y_predict = reg.predict(X_test)

# GET IMPORTANT VALUES
coefs = model.coef_

# summarize feature importance
# for i, v in enumerate(coefs):
#     print('Feature: %0d, Score: %.5f' % (i, v))

# PLOT IMPORTANT FEATURES
plt.figure(1)
plt.subplot(111)
plt.bar([x for x in range(len(coefs))], coefs)
plt.xlabel("Features")
plt.ylabel("Importance Values")

# PLOT IMPORTANT FEATURES AGAINST OUTPUT VARIABLE
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, [5]], X_train[:, [6]], Y_train, marker='.', color='red')
ax.set_xlabel("Coarse Aggregate")
ax.set_ylabel("Fine Aggregate")
ax.set_zlabel("Concrete Strength ")

# PLOT MODEL EVALUATION
plt.figure(3)
plt.subplot(111)
ax1 = sns.distplot(Y_test, hist=False, color="r", label="Actual Value")
sns.distplot(Y_predict, hist=False, color="b", label="Fitted Values", ax=ax1)

plt.show()

# MODEL EVALUATION
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_predict})
print(df)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_predict))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)))
