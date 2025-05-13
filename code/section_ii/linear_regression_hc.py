import numpy as np
import pandas as pd


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "{0},{1}".format(self.x, self.y)


points = [(Point(row.x, row.y)) for index, row in pd.read_csv("https://bit.ly/2KF29Bd").iterrows()]

# Building the model
m = 0.0
b = 0.0

epochs = 150000  # The number of iterations to perform

n = float(len(points))  # Number of points

best_loss = 10000000000000.0  # Initialize with a really large value

for i in range(epochs):

    # Randomly adjust "m" or "b"

    m_adjust = np.random.normal()
    b_adjust = np.random.normal()

    m += m_adjust
    b += b_adjust

    # Calculate loss, which is total sum squared error
    new_loss = 0.0
    for p in points:
        new_loss += (p.y - (m * p.x + b)) ** 2

    # If loss has improved, keep new values. Otherwise revert.
    if new_loss < best_loss:
        print("y = {0}x + {1}".format(m, b))
        best_loss = new_loss
    else:
        m -= m_adjust
        b -= b_adjust

print("y = {0}x + {1}".format(m, b))


# plot
import matplotlib.pyplot as plt
import numpy as np
# show in chart
X = np.array([p.x for p in df.itertuples()])
Y = np.array([p.y for p in df.itertuples()])

plt.plot(X, Y, 'o') # scatterplot
plt.plot(X, m*X+b) # line
plt.show()
