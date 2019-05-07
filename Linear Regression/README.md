# Linear Regression

Linear regression algorithm is mainly used in regression problems, which means the value of result is continuous instead of discrete.
It aims at finding a best straight line crossing the dataset, together with a minimal MSE. 

<img width='405' height='300' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Linear%20Regression/data/regression_line.png"/>

## Standard linear regression

Assume that we use `MSE` as the cost function, and then we can optimize our model in the following ways: 


- Gradient Descent

Use the formulars below, we can easily get all the partial derivatives, and then feed them to gradient descent:

<img width='950' height='217' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Linear%20Regression/data/linear_formular.png"/>

Then the total error will decrease after every iteration:

<img width='405' height='300' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Linear%20Regression/data/learning_curve.png"/>


- Normal Equation

In addition, we can also get the optimal weights directively, by using `Normal Equation`.

<img width='150' height='30' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Linear%20Regression/data/normal_equation.png"/>


## Local weighted linear regression (LWLR)
To avoid general underfit, LWLR includes `local weights W` to measure the distance between input and training example. `W` is a diagonal  matrix, while the value at position(i,i) measures the distance between input and the i-th sample. It will increase as the distance decreases. Using the formulars below, we can get the weights for LWLR:

<img width='180' height='83' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Linear%20Regression/data/LWLR.png"/>


The regression line turns out as follow:

<img width='405' height='300' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Linear%20Regression/data/lwlr.png"/>

