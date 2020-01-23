import pandas as pd
from sklearn.linear_model import LinearRegression

# Learn more: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# Import points
points = pd.read_csv("https://bit.ly/2KF29Bd")
x = points.iloc[:, 0:1]
y = points.iloc[:, 1:2]

print(y)

# Plain ordinary least squares
fit = LinearRegression().fit(x, y)

# Print "m" and "b" coefficients
print("m = {0}".format(fit.coef_.flatten()))
print("b = {0}".format(fit.intercept_.flatten()))

# Predict a new "y" value for x = 3.5
print("x = 3.5, y = {0}".format(fit.predict([[3.5]]).flatten()))
