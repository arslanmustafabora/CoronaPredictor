import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

### IMPORT DATA ###
data = pd.read_csv('total_cases.csv',sep=',')

### PREPROCESS DATA ###
data = data[['date','Turkey']]
data[['date']] = np.arange(len(data[['date']]))+1
data[['Turkey']] = data[['Turkey']].fillna(0)
x = np.array(data['date']).reshape(-1,1)
y = np.array(data['Turkey']).reshape(-1,1)

### TRAINING ###
polynomialFeatures = PolynomialFeatures(degree=6)
x = polynomialFeatures.fit_transform(x)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print("Accuracy of prediction will be: ",end="")
print(round(accuracy*100,2))
y0 = model.predict(x)

### VISUALIZATION ###
plt.plot(y,'-m')
plt.plot(y0,'--b')
plt.show()

### PRINTING OUT THE SAMPLE ###
length = len(data[['date']])
days = 2
print(f'Prediction after {days} days: ',end='')
print(int(model.predict(polynomialFeatures.fit_transform([[length+days]]))))

