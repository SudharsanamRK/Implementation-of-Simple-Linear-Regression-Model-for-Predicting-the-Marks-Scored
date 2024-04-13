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

# Load the data
df = pd.read_csv('/content/Data.csv')

# Split data into features (x) and target (y)
x = df[['x']]
y = df['y']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the Linear Regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Plot the scatter plot of data points
plt.scatter(df['x'], df['y'])
plt.xlabel('x')
plt.ylabel('y')

# Plot the regression line
plt.plot(x_train, lr.predict(x_train), color='red')

# Show the plot
plt.show()

# Coefficient And Intercept Values
lr.coef_
lr.intercept_
```

## Output:
## 1) Head:
![image](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/6b59b3e2-01c6-4c0e-b3dd-4c3d789d4f49)

## 2) Graph Of Plotted Data:
![image](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/2728b20d-789a-42a9-aa87-fbbf4cafa8bb)

## 3) Trained Data:
![image](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/dac31c12-c153-4497-a476-d360590a7ff0)

## 4) Line Of Regression:
![image](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/0860b80a-f747-49f2-916c-3ccb440cf8f7)

## 5) Coefficient And Intercept Values:
![image](https://github.com/SudharsanamRK/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115523484/c418f0d5-6ae5-4f11-88d2-f2c84f7de37b)


 
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
