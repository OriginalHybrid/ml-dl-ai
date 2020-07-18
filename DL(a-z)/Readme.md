### Deep Learning Models
----------------------------------------

#### ANN

##### Usecase:
Churm Modelling Problem
Dataset:
A Dataset of bank customers with their details and the decision whether they will leave the bank or not.
(can also be applied to like should the customer get the loan or credit card approval etc)

---------------------------------------------
#### RNN

##### Usecase:

Predict the future pattern trend of google stock price using LSTM.

**Approach:**

Train LSTM on 5 years of stock price data and based on the training and corelation identified or captured by LSTM of Google stock price we will try to predict the first month of 2017, Jan.
WE will not exactly predict the stock price, we will predict the upward or downward trend of google stock price


--------------------------------------------

#### SOP

##### Usecase: 

Fraud Detection

**Approach:**

Given Dataset containing the information of customers of the bank applying for and advanced credit card.
Detect Potential fraud within these applications
Give the explicit list of customers who potentially cheated
We will detect patterns using unsupervised deep learning in a high dimensional dataset full of non linear relationships
and one of these patterns will be the potential fraud

**Data Set Information:**

This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

This dataset is interesting because there is a good mix of attributes -- continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.

**Attribute Information:**

There are 6 numerical and 8 categorical attributes. The labels have been changed for the convenience of the statistical algorithms. For example, attribute 4 originally had 3 labels p,g,gg and these have been changed to labels 1,2,3.

A1: 0,1 CATEGORICAL (formerly: a,b)
A2: continuous.
A3: continuous.
A4: 1,2,3 CATEGORICAL (formerly: p,g,gg)
A5: 1, 2,3,4,5, 6,7,8,9,10,11,12,13,14 CATEGORICAL (formerly: ff,d,i,k,j,aa,m,c,w, e, q, r,cc, x)
A6: 1, 2,3, 4,5,6,7,8,9 CATEGORICAL (formerly: ff,dd,j,bb,v,n,o,h,z)
A7: continuous.
A8: 1, 0 CATEGORICAL (formerly: t, f)
A9: 1, 0 CATEGORICAL (formerly: t, f)
A10: continuous.
A11: 1, 0 CATEGORICAL (formerly t, f)
A12: 1, 2, 3 CATEGORICAL (formerly: s, g, p)
A13: continuous.
A14: continuous.
A15: 1,2 class attribute (formerly: +,-)


---------------------------------------------------

#### Boltzmann Machine and Auto Encoder

##### Usecase:
 
Build a  recommender systems
One that will predict a binary outcome yes/no whether the user will like the movie or not.
And other that will predict the rating.
Using both Boltzman and Auto Encoder
