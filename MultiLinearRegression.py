import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# style.use('ggplot')

# RETRIEVE TRAINING DATA
data = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')
X_in = data.iloc[:, 0:8]  # Get the first eight columns of data
Y_in = data.iloc[:, -1]

# STANDARDIZE DATA
sc = StandardScaler()
X_in = sc.fit_transform(X_in)

# INITIALIZE COEFFICIENTS
learning_rate = 0.005
iterations = 10000
B = np.zeros(X_in.shape[1])  # Initialize values of thetas


# COST MSE
def cost_function(X, Y, B):
    n = len(Y)
    return np.sum((X.dot(B) - Y) ** 2) / (2 * n)


initial_cost = cost_function(X_in, Y_in, B)
print("Ininital_cost", initial_cost)


def gradient_descent(X, Y, B_cur):
    n = len(Y)
    cost_data = []
    itr = []
    for i in range(iterations):
        y_hypo = X.dot(B_cur)
        gradient = X.T.dot(y_hypo - Y) / n
        B_cur = B_cur - learning_rate * gradient
        cost = cost_function(X, Y, B_cur)
        cost_data.append(cost)
        itr.append(i)

    return B_cur, cost_data, itr


# SPLITTING DATASET TO TRAINING AND TESTING SETS
X_train, X_test, Y_train, Y_test = train_test_split(X_in, Y_in, test_size=0.2, random_state=0)

b, cost_arr, iteration_nums = gradient_descent(X_train, Y_train, B)
# print(b)
print("Learning cost: ", cost_arr[-1])


# MODEL EVALUATION
def r_mse(Y, Y_predict):
    return np.sqrt(sum((Y - Y_predict) ** 2) / len(Y))


Y_pred = X_test.dot(b)
rmse = r_mse(Y_test, Y_pred)
print("RMSE: ", rmse)
# CREATE PLOT FIGURES
fig, ax = plt.subplots(1)  # Cost figure
# fig2, ax2 = plt.subplots(1)  # Data figure

# fig2 = plt.figure()
# ax2 = Axes3D(fig2)
# ax2.scatter(X_in[0], X_in[1], Y_in, color='#ef1234')

ax.plot(iteration_nums, cost_arr)
plt.show()
