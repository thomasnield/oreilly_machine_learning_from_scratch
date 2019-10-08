
## ==================================================
## Using Hill Climbing
## ==================================================


import pandas as pd

from random import normalvariate

# Input data
data = pd.read_csv('https://bit.ly/2UBhrMG', header=None)
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Building the model
a = 0.0
b = 0.0

epochs = 20000  # The number of iterations to perform

n = float(len(X))  # Number of elements in X

# Fit to function f(x) = ax^2 + b by solving for "a" and "b"
# Performing hill climbing

best_loss = 10000000000000.0 # Initialize with a really large value 

for i in range(epochs):

	#Randomly adjust "a" and "b" 
    a_adjust = normalvariate(0,1) # can also use t-distribution from NumPy, SciPy
    b_adjust = normalvariate(0,1) # can also use t-distribution from NumPy, SciPy
    a += a_adjust
    b += b_adjust

	# Calculate loss, which is total mean squared error
    new_loss =  (1/n) * sum((Y - (a * X**2 + b))**2)  # The current predicted value of Y

	# If loss has improved, keep new values. Otherwise revert. 
    if new_loss < best_loss:
        best_loss = new_loss
    else:
        a -= a_adjust
        b -= b_adjust


print(a, b)
