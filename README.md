# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5. Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sudharsanam R K
RegisterNumber: 212222040163
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/MLSET.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```

## Output:
## 1) Head:
![head](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/46b8c517-4fff-4a20-8126-acc77ff9fb0a)
## 2) Graph Of Plotted Data:
![plotted_data](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/6b009754-9c2d-4c79-a5c6-9bc955a6d789)
## 3) Trained Data:
![trained_data](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/2ae3c06d-c5b3-4dec-8991-0394740e2406)
## 4) Line Of Regression:
![l_o_r](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/c564dcae-3ebb-47bd-a723-322f3da62293)
## 5) Coefficient And Intercept Values:
![Coefficient-And-Intercept-Values](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/ed77cee8-c74f-418b-b534-10bbf8a9a7fb)


 
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
