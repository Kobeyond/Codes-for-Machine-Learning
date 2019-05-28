# Naive Bayes
Naive bayes is a useful classfication algorithm in machine learning. It aims at computing the probablities of every class when given the input x, and then choose the class label with the largest probablity as the final result. 

It's based on applying `Bayes' theorem` with strong (naive) independence assumptions between the features. In other words, we assume that all the features are independent to each other.

## Bayes' Theorem

Suppose x=(x1, x2, ..., xn) is the input, and Ci is the label of the i-th class. So, we can compute the conditional probs as follow:

 <img width='280' height='69' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Naive%20Bayes/data/bayes.png"/>

As we know, the probability of simultaneous occurrences based on independent events is equal to the product of the probability of each event occurring alone. In naive bayes, we assume all the features are independent to each other, so the probablity of P(x|Ci) can be computed as follow:

 <img width='700' height='70' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Naive%20Bayes/data/bayes2.png"/>

## Example1: Judging Insulting Comments

Given series of labeled (insulting or not) comments from website, train a bayes model to judge whether a comment is insulting. The dataset turns out like this:

 <img width='440' height='213' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Naive%20Bayes/data/posts.png"/>


## Example2: Spam or Ham

Given many labeled (spam or ham) emails as dataset, train a bayes model to judge whether a email is spam or ham. In both of the two problems, we need to create a `vocabulary` based on dataset, and then convert the text to a fixed-length vector (one-hot vector).

