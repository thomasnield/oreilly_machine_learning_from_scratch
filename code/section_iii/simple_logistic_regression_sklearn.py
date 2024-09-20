import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("https://tinyurl.com/y2cocoo7")

# grab independent variable column
inputs = data.iloc[:, :-1]

# grab dependent variable column
output = data.iloc[:, -1]

# build logistic regression, note CVLogisticRegression is also recommended to use cross-validation
fit = LogisticRegression().fit(inputs, output)

# Print coefficients:
print("COEFFICIENTS: {0}".format(fit.coef_.flatten()))
print("INTERCEPT: {0}".format(fit.intercept_.flatten()))

# Test a prediction
print("x=1.5, y={0}".format(fit.predict([[1.5]])))
print("x=18.5, y={0}".format(fit.predict([[18.5]])))
