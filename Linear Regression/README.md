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

formular here!!

## Local weighted linear regression (LWLR)
To avoid general underfit, LWLR includes `local weights` W to measure the distance between input and every example. 
formular here!!

<img width='405' height='300' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Linear%20Regression/data/lwlr.png"/>

