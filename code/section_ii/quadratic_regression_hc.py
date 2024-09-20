import pandas as pd
from random import uniform, normalvariate, randint


# Generate random data around y = 4(x-3)^2 + 50 
# Desmos Graph: https://www.desmos.com/calculator/cdk3tndz6d

def f(x):
    return 4 * (x - 3) ** 2 + 50

data = pd.DataFrame(columns=["x", "y"])

for i in range(1000):
    x = round(uniform(1, 20), 2)
    data = data.append(pd.DataFrame(columns=["x", "y"], data=[[x, round(f(x) + normalvariate(0, 50), 2)]]))

# Input data
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Building the model
a = 0.0
b = 0.0
c = 0.0

epochs = 1000000  # The number of iterations to perform

n = float(len(X))  # Number of elements in X

# f(x) = a(x-b)^2 + c

best_loss = 10000000000000.0  # Initialize with a really large value

for i in range(epochs):

    change_var_index = randint(1, 3)

    # Randomly adjust "a","b", or "c"
    adjust = normalvariate(0, 1)  # can also use t-distribution from NumPy, SciPy: np.random.standard_t(3, 1)[0]

    if change_var_index == 1:
        a += adjust
    if change_var_index == 2:
        b += adjust
    if change_var_index == 3:
        c += adjust

    # Calculate loss, which is total mean squared error
    new_loss = (1 / n) * sum((Y - (a * (X - b) ** 2 + c))**2 )

    # If loss has improved, keep new values. Otherwise revert.
    if new_loss < best_loss:
        best_loss = new_loss
    else:
        if change_var_index == 1:
            a -= adjust
        if change_var_index == 2:
            b -= adjust
        if change_var_index == 3:
            c -= adjust

    if i % 1000 == 0:
        print(a, b, c)

print(a, b, c)
