# LinearRegression
Implementing the gradient descent algorithm, and apply it to predict the Concrete compressive strength (MPa, megapascal)

Google Colab: Run the code without any requirements.
IDE: Pycharm (Recommend)

In order to successfully compile and run the program. External libraries including pandas, matplotlib, sklearn, seaborn need to be installed.

“ pip install pandas”.
“pip install matplotlib”
“pip install sklearn”.
“pip install seaborn”.

Adjust learning_rate and Iterations parameters and run the code until the Cost value is minimum.
The log file named “log.txt” contains the parameters have been used as well as the Cost of each set of parameters.

In the first case, when the learning rate is too small (0.08) combine with few iterations (1000). The cost is really high (> 111).
However, when the iteration is increased. The cost also decrease quickly until the learning rate is set at the value around 0.8 and 0.9. With the high learning rate as well as iterations, the Cost has a tendency to decrease at a low rate and maintain the value around 55.
Moreover, we evaluate the model by the Root Mean Square Error comparing the “Actual Values” and “Predict Values”. The result maintains below 10 implicates the model works well.
As a result of which, the chosen parameters are good enough to provide a satisfied result.

<img src="https://user-images.githubusercontent.com/54776410/105113668-f528a380-5a8a-11eb-892d-78f9137b4f4f.png">

<img src="https://user-images.githubusercontent.com/54776410/105113781-3b7e0280-5a8b-11eb-81e7-7a16c394e722.png">
