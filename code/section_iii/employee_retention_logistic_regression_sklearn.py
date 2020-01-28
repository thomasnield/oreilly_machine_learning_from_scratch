import pandas as pd
from sklearn.linear_model import LogisticRegression

employee_data = pd.read_csv("https://tinyurl.com/y6r7qjrp")

# grab independent variable columns
inputs = employee_data.iloc[:, :-1]

# grab dependent "did_quit" variable column
output = employee_data.iloc[:, -1]

# build logistic regression
fit = LogisticRegression().fit(inputs, output)

# Print coefficients:
print("COEFFICIENTS: {0}".format(fit.coef_.flatten()))
print("INTERCEPT: {0}".format(fit.intercept_.flatten()))

print(output)
print(fit.predict(inputs))

# Test a predictions
while True:
    n = input("Predict employee will stay or leave {sex},{age},{promotions},{years employed}: ")
    (sex, age, promotions, years_employed) = n.split(",")
    print(fit.predict([[int(sex), int(age), int(promotions), int(years_employed)]]))

