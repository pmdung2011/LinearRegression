import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# RETRIEVE TRAINING DATA
data = pd.read_excel('https://datasetdanny.s3.amazonaws.com/train.xls')
X_in = data.iloc[:, 0]
X_in = (X_in - np.mean(X_in)) / np.std(X_in)  # Normalize data
# X_in = np.c_[np.ones(X_in.shape[0]), X_in]
Y_in = data.iloc[:, 8]

# def cost_function(X, y, theta):
#     m = len(y)
#     o = pd.DataFrame(np.ones(m))
#     squared_err = (np.dot(X, theta) - y) ** 2
#     return np.dot(o.T, squared_err) / (2 * m)

# CREATE PLOT FIGURES
fig, ax = plt.subplots(1)  # Cost figure
fig2, ax2 = plt.subplots(1)  # Data figure


# FUNCTION TO CALCULATE PERCENTAGE DIFF
def percentage_different(a, b):
    if a - b == 0:
        return 0
    else:
        return abs(a - b) * 100


# GRADIENT DESCENT
def gradient_descent(x_input, y_input):
    m_cur = 0
    b_cur = 0
    iterations = 1000
    n = len(y_input)
    learning_rate = 0.01
    iter = []
    cost_arr = []
    for i in range(iterations):
        y_hypothesis = m_cur * x_input + b_cur  # y= mx+b
        cost = ((1 / n) * sum((y_input - y_hypothesis) ** 2))
        m_d = -(2 / n) * sum(x_input * (y_input - y_hypothesis))  # derivative of m
        b_d = -(2 / n) * sum(y_input - y_hypothesis)  # derivative of b
        m_cur = (m_cur - learning_rate * m_d)  # update m
        b_cur = (b_cur - learning_rate * b_d)  # update b
        iter.append(i)
        cost_arr.append(cost)

    return m_cur, b_cur, cost_arr, iter


# INVOKE GRADIENT DESCENT
m, b, cost, itr = gradient_descent(X_in, Y_in)

# Plot the cost function...
ax.plot(itr, cost)

# Plot the data and regression line
regression_line = [(m * x) + b for x in X_in]
ax2.scatter(X_in, Y_in)
ax2.plot(X_in, regression_line, "g")
print("m {}, b {}, cost {}".format(m, b, cost[len(cost) - 1]))
plt.show()


# RETRIEVE DATA FOR TESTING
data_test = pd.read_excel('https://datasetdanny.s3.amazonaws.com/test.xls')
X_test = data.iloc[:, 0]
X_test = (X_test - np.mean(X_test)) / np.std(X_test)
# X_in = np.c_[np.ones(X_in.shape[0]), X_in]
Y_test_value = np.array(data.iloc[:, 8])

# PREDICT Y OUTPUT
Y_predict = [(m * x) + b for x in X_test]

# ax2.plot(X_test, Y_predict, "b")
df = pd.DataFrame({'Actual': Y_test_value.flatten(), 'Predicted': np.array(Y_predict).flatten()})
print(df)
# Calculate the precision of the algorithm
cost_value = ((1 / len(Y_predict)) * sum((Y_predict - Y_test_value) ** 2))
print("Regression Line: Y = ", m, "X + ", b)
print("Percentage different: ", percentage_different(cost_value, cost[len(cost) - 1]), "%")
