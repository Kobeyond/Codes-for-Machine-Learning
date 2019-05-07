# Logistic Regression

`Logistic Regression` algorithm is mainly used to binary classification, which aims at finding a best decision boundary
to separate the training examples. While z >= 0 means positive (above the line), and z <= 0 means negative(beyond the
line). In addition, g(z) represents the probability of positive.

Assume we use `sigmod` as activation function and `cross entropy` as cost function, then we can get all the partial derivatives as below:

<img width='950' height='302' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Logistic%20Regression/data/logistic_formulars.png"/>

It's quite a coincidence that the result of logistic regression is actually the same as linear regression! Now, using the `vectorized formular` above, we can easily train our logistic model.



## Horse Colic classification
Given various symptoms of a horse, we use logistic regression to predict whether the horse is suffering colic(1 or 0).
<img width='450' height='329' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Logistic%20Regression/data/logistic.png"/>

