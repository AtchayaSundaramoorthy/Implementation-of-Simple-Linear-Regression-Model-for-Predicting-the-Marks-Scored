# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Plot the independent variable X and dependent variable Y
3. predict the values
4. plot the predicted values 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ATCHAYA S
RegisterNumber: 212222040021 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ML - Sheet1.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')
lr.coef_
lr.intercept_

```

## Output:
1)HEAD:

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393516/064cafb6-3d51-4565-9b7d-050ce3bc848d)

2)GRAPH OF PLOTTED DATA:

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393516/c62c89a9-8085-4636-8e00-3cb841ffc87a)

3)TRAINED DATA

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393516/23ccdf12-5959-4b98-94f2-d6459de4cf6e)

4)LINE OF REGRESSION:

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393516/c898f6a3-5d12-430b-9995-abd2a78797bd)

5)COEFFICIENT AND INTERCEPT VALUES:

![image](https://github.com/AtchayaSundaramoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393516/7490163c-3ec2-43fa-a582-3f65ba402412)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
