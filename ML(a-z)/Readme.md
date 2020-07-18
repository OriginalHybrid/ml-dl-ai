### Machine Learning Models
----------------------------------------

#### Simple Linear Regression

##### Usecase:

Given a dataset of Years of Experience and Salary
Find if there is any sort of corelation between the two.

Helps to set and find a relation for years and salary and then later for say helps to find the estimate salary that the company
would give to a person with a no of years experience.

-----------------------------------------

#### Multiple Linear Regression

##### Usecase:

Given a dataset of 50 startups with their expenditure in different departments and annual profit.
Analyze the dataset and create a model which will tell the desirable company to invest based on profit.
Understanding the areas of accessment that which company performs better ie based on location, marketing or rnd or administrarion
with the goal to maximize the profit.

----------------------------------------

#### Polynomial Regression/ Support Vector Regression

##### Usecase:

Given a dataset of Position, Level and Salary
Find if there is any sort of corelation between them.

Helps to set and find a relation for Position, level and salary and then later for say helps to find the estimate salary that the company
would give to a person with a particular level or position.
OR test using the model whether the canditate is rightly negotiating the salary for his level.

----------------------------------------

#### Decision Tree Regression and Random Forest Regression

##### Usecase:

Given a dataset of Position, Level and Salary
Find if there is any sort of corelation between them.

Helps to set and find a relation for Position, level and salary and then later for say helps to find the estimate salary that the company
would give to a person with a particular level or position.
OR test using the model whether the canditate is rightly negotiating the salary for his level.

----------------------------------------

#### Logistic Regression/ KNN/ Naive Bayes/ Decision Tree Classification/ SVM/ Random Forest Classification

##### Usecase:

Given a dataset which contains information of users of a social network.(UserID, Gender, Age, Estimated Salary)
This social network has various clients which put their ads on it, one of 
which is a car company which has recently released a new SUV.
We intend to find out which of these users will be interested in buying this SUV

Last Column of dataset (yes/no) tells whether the user has bought the SUV or not. 

----------------------------------------

#### K Means Clustering/ Heirarchical Clustering 

##### Usecase:

Given a Mall dataset which contains information of users of customers having a membership card.(CustomerID, Gender, Age, Annual Income, Spending Score(1-100)
Segment the customers into two different groups based on Annual Income and Spending Score 

----------------------------------------

#### Apriori Model/ Elcat Algo

##### Usecase:

Optimising sales in a grocery store.
Using Association Rule learning, to know where to place the product Stores optimize the sales of ther product.
Eg. If someone buys cerealsm the same person is very likely to buy mils as well, so place milk near cereals.
Products which are more associated should be placed next to each other so that customer sub consiously buy both.

Dataset contains 7500 transactions of different customers and columns contains different products.

----------------------------------------

#### Upper Confidence Bound/ Thompson Sampling

##### Usecase:
Solving Multi Arm Bandit Problem:
In probability theory, the multi-armed bandit problem (sometimes called the K-[1] or N-armed bandit problem) is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice. This is a classic reinforcement learning problem that exemplifies the exploration–exploitation tradeoff dilemma. The name comes from imagining a gambler at a row of slot machines (sometimes known as "one-armed bandits"), who has to decide which machines to play, how many times to play each machine and in which order to play them, and whether to continue with the current machine or try a different machine.
We are optimising the CTR(click through rate) of diiferent users on a ad that we put on social network.

Dataset contains columns of different ads and rows of different users where cell contains if the ad has been clicked.

Useful in changing strategy to place the ads in social network(according to the results observed)
ie according to results oberved the algo will decide which version of ad it will show to the user.

----------------------------------------

#### PCA/ LDA/ Kernel PCA

##### Usecase:
DIMENSIONALITY REDUCTION
From the m indeoendent variable of your dataset, PCA extracts p <= m new indeoendent variables
that explain the moset variance of the dataset.

Dataset:
It is a Wine dataset where column contains different feature imformation of the wine.
It contains n-1 columns of independent variables nad the last column as customer segment as dependent variable.
Approach:
Take all info of wine and customer segment and make a classification model like logistic regression so that it can predict for 
each new wine which customer segment it shoud refer to.
Since it is nearly impossible to plot 13 dimensions, we will extract 2 variables
usong dimensionality reduction that explain the most of the variance.

----------------------------------------

#### XGBoost

##### Usecase:
Churm Modelling Problem
Dataset:
A Dataset of bank customers with their details and the decision whether they will leave the bank or not.