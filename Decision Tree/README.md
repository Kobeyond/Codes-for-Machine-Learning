# Decision Tree
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences.
When predicting an answer, you will go deeper and deeper until arriving at a leaf node. Finally, the info of the leaf node is the answer.

## Shannon Entropy and Info Gain
Shannon entropy means the uncertainty or confusion of a dataset, and Info gain means the change of shannon entropy in a dataset.
 
 <img width='300' height='71' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Decision%20Tree/data/pictures/formulars.png"/>


It turns out that when splitting a large dataset to smaller ones, the feature which causes largest info gain is the best choice.
In other words, selecting this feature as axis to split the orginal dataset is optimal. 

## Example: Lenses Judgement
Given a series of information about the patient(age, tear rate and so on), predict whether the patient is appropriate to use lenses.

<img width='1100' height='489' src="https://github.com/Kobeyond/Codes-for-Machine-Learning/blob/master/Decision%20Tree/data/tree.png"/>

