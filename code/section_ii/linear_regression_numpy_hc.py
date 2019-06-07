import numpy as np
import pandas as pd

data = pd.read_csv("https://tinyurl.com/y58sesrr")

# Input data
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Building the model
m = 0.0
b = 0.0


epochs = 150_000  # The number of iterations to perform

n = float(len(X))  # Number of elements in X


best_loss = 10000000000000.0  # Initialize with a really large value

for i in range(epochs):

    # Randomly adjust "m" or "b"

    m_adjust = np.random.standard_t(3, 1)[0] #or use normalvariate(0, 1)
    b_adjust = np.random.standard_t(3, 1)[0] #or use normalvariate(0, 1)

    m += m_adjust
    b += b_adjust

    # Calculate loss, which is total mean squared error
    new_loss = (1 / n) * sum((Y - (m * X + b))**2 )

    # If loss has improved, keep new values. Otherwise revert.
    if new_loss < best_loss:
        print("y = {0}x + {1}".format(m, b))
        best_loss = new_loss
    else:
        m -= m_adjust
        b -= b_adjust

print("y = {0}x + {1}".format(m,b))
