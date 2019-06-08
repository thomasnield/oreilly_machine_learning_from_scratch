
import random

import numpy as np
import pandas as pd

iterations = 1_000_000
employee_data = pd.read_csv("https://tinyurl.com/y6r7qjrp")


# extract independent variables (SEX,AGE,PROMOTIONS,YEARS_EMPLOYED) for all records
training_inputs = employee_data.iloc[:,0:4]

# append a column of 1.0 for the b0 value
training_inputs.insert(loc=0,column='b0',value=1.0)
training_inputs = training_inputs.values

# extract dependent variables (DID_QUIT) for all records
training_outputs = employee_data.iloc[:,4:5].values.flatten()

# track best_likelihood and beta values, which we will use hill-climbing to solve for
best_likelihood = -100_000_000_000.0
betas = np.array([1.0,1.0,1.0,1.0,1.0])

# calculate maximum likelihood
for i in range(iterations):

    # Select b0, b1, b2, b3, or b4 randomly, and adjust it by a random amount
    random_b = random.choice(range(5))
    random_adjust = np.random.standard_normal()
    betas[random_b] += random_adjust

    # calculate new likelihood
    # Use logarithmic addition to avoid multiplication and decimal underflow

    probabilities = 1.0 / (1.0 + np.exp(-1.0 * training_inputs.dot(betas)))

    true_likelihood = (np.log(probabilities) * training_outputs).sum()
    false_likelihood = (np.log(1.00001 - probabilities) * ((training_outputs - 1) **2)).sum()

    new_likelihood = true_likelihood + false_likelihood

    if new_likelihood > best_likelihood:
        best_likelihood = new_likelihood
    else:
        betas[random_b] -= random_adjust

# Print best result
print("1.0 / (1 + exp(-({0} + {1}*s + {2}*a + {3}*p + {4}*y))".format(betas[0],betas[1],betas[2],betas[3],betas[4]))
print("BEST LIKELIHOOD: {0}".format(np.math.exp(best_likelihood)))


# Interact and test with new employee data

def predict_probability(sex, age, promotions, years_employed):
    x = -(betas[0] + (betas[1] * sex) + (betas[2] * age) + (betas[3] * promotions) + (betas[4] * years_employed))
    odds = np.math.exp(x)
    p = 1.0 / (1.0 + odds)
    return p


def predict_employee_will_stay(sex, age, promotions, years_employed):
    probability_of_leaving = predict_probability(sex, age, promotions, years_employed)
    if probability_of_leaving >= .5:
        return "WILL LEAVE, {0}% chance of leaving".format(round(probability_of_leaving * 100.0,2))
    else:
        return "WILL STAY, {0}% chance of leaving".format(round(probability_of_leaving * 100.0,2))


while True:
    n = input("Predict employee will stay or leave {sex},{age},{promotions},{years employed}: ")
    (sex, age, promotions, years_employed) = n.split(",")
    print(predict_employee_will_stay(int(sex), int(age), int(promotions), int(years_employed)))

