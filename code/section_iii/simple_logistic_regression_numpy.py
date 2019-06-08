import random

import math
import numpy as np
import pandas as pd

iterations = 1_000_000
employee_data = pd.read_csv("https://tinyurl.com/y2cocoo7")


# extract independent variables (SEX,AGE,PROMOTIONS,YEARS_EMPLOYED) for all records
training_inputs = employee_data.iloc[:,0:1]

# append a column of 1.0 for the b0 value
training_inputs.insert(loc=0,column='b0',value=1.0)
training_inputs = training_inputs.values

# extract dependent variables (DID_QUIT) for all records
training_outputs = employee_data.iloc[:,1:2].values.flatten()

# track best_likelihood and beta values, which we will use hill-climbing to solve for
best_likelihood = -100_000_000_000.0
betas = np.array([1.0,1.0])


# calculate maximum likelihood

def predict_probability(x):
    p = 1.0 / (1.0 + math.exp(-(betas[0] + betas[1] * x)))
    return p


# calculate maximum likelihood
for i in range(iterations):

    # Select b0 or b1 randomly, and adjust it by a random amount
    random_b = random.choice(range(2))
    random_adjust = np.random.standard_normal()
    betas[random_b] += random_adjust

    # calculate new likelihood
    # Use logarithmic addition to avoid multiplication and decimal underflow

    probabilities = 1.0 / (1.0 + np.exp(-1.0 * training_inputs.dot(betas)))
    probabilities2 = [predict_probability(x1) for (x0,x1) in training_inputs]

    true_likelihood = (np.log(probabilities) * training_outputs).sum()
    false_likelihood = (np.log(1.000001 - probabilities) * ((training_outputs - 1)**2)).sum()

    new_likelihood = true_likelihood + false_likelihood

    if new_likelihood > best_likelihood:
        best_likelihood = new_likelihood
    else:
        betas[random_b] -= random_adjust

print("1.0 / (1 + exp(-({0} + {1}*x))".format(betas[0], betas[1]))
print("BEST LIKELIHOOD: {0}".format(best_likelihood))
