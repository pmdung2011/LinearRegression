import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns

# RETRIEVE TRAINING DATA
df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')
X_in = df.iloc[:, 0:8]  # Get the features
Y_in = df.iloc[:, -1]

# NORMALIZE DATA
X_in = preprocessing.normalize(X_in, norm='l2')

# INITIALIZE COEFFICIENTS
learning_rate = 0.8
iterations = 10000
B = np.zeros(X_in.shape[1])  # Initialize values of thetas


# COST MSE
def cost_function(X, Y, B):
    n = len(Y)
    predictions = X.dot(B).flatten()
    return (1.0 / (2 * n)) * ((predictions - Y) ** 2).sum()


initial_cost = cost_function(X_in, Y_in, B)
print("Initial_cost_MSE", initial_cost)


# GRADIENT DESCENT FUNCTION
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
print("Learning cost_MSE: ", cost_arr[-1])


# MODEL EVALUATION ROOT MEAN SQUARED ERROR
def r_mse(Y, Y_predict):
    return np.sqrt(sum((Y - Y_predict) ** 2) / len(Y))


# SAVE PARAMETERS INTO LOG FILE
def log_file(steps, itr, c):
    file = open("log.txt", "a")
    file.write("Learning_rate: {} \n".format(str(steps)))
    file.write("Iterations: {} \n".format(str(itr)))
    file.write("Cost: {} \n".format(str(c)))
    file.close()


log_file(learning_rate, iterations, cost_arr[-1])

Y_pred = X_test.dot(b)
rmse = r_mse(Y_test, Y_pred)
print("Evaluate Model_RMSE: ", rmse)

# CREATE PLOT FIGURES
plt.figure(1)
plt.plot(iteration_nums, cost_arr)
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.figure(2)
plt.subplot(111)
ax1 = sns.distplot(Y_test, hist=False, color="r", label="Actual Value")
sns.distplot(Y_pred, hist=False, color="b", label="Fitted Values", ax=ax1)

plt.show()
