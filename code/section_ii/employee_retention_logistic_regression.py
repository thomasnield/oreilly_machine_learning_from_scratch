import random

import math
import pandas as pd
import numpy as np
from numpy import log, exp


class EmployeeRetention:
    def __init__(self, sex, age, promotions, years_employed, did_quit):
        self.sex = sex
        self.age = age
        self.promotions = promotions
        self.years_employed = years_employed
        self.did_quit = did_quit


training_data = [(EmployeeRetention(row[0], row[1], row[2], row[3], row[4])) for index, row in
                 pd.read_csv("https://tinyurl.com/y6r7qjrp").iterrows()]


best_likelihood = -100_000_000_000.0
b0 = 1.0  # constant
b1 = 1.0  # sex beta
b2 = 1.0  # age beta
b3 = 1.0  # promotions beta
b4 = 1.0  # years employed beta

iterations = 50_000


# calculate maximum likelihood

# Closer to true (1.0) recommends dark font, closer to false (0.0) recommends light font
def predict_probability(sex, age, promotions, years_employed):
    x = -(b0 + (b1 * sex) + (b2 * age) + (b3 * promotions) + (b4 * years_employed))
    odds = exp(x)
    p = 1.0 / (1.0 + odds)
    return p


for i in range(iterations):

    # Select b0, b1, b2, or b3 randomly, and adjust it by a random amount
    random_b = random.choice(range(5))

    random_adjust = np.random.standard_normal()

    if random_b == 0:
        b0 += random_adjust
    elif random_b == 1:
        b1 += random_adjust
    elif random_b == 2:
        b2 += random_adjust
    elif random_b == 3:
        b3 += random_adjust
    elif random_b == 4:
        b4 += random_adjust

    # calculate new likelihood
    # Use logarithmic addition to avoid multiplication and decimal underflow
    new_likelihood = 0.0

    for emp in training_data:

        probability = predict_probability(emp.sex, emp.age, emp.promotions, emp.years_employed)

        if emp.did_quit == 1:
            new_likelihood += log(probability)
        else:
            new_likelihood += log(1.0 - probability)

    # If solution improves, keep it and make it new best likelihood. Otherwise undo the adjustment
    if best_likelihood < new_likelihood:
        best_likelihood = new_likelihood
    elif random_b == 0:
        b0 -= random_adjust
    elif random_b == 1:
        b1 -= random_adjust
    elif random_b == 2:
        b2 -= random_adjust
    elif random_b == 3:
        b3 -= random_adjust
    elif random_b == 4:
        b4 -= random_adjust

# Print best result
print("1.0 / (1 + exp(-({0} + {1}*s + {2}*a + {3}*p + {4}*y))".format(b0, b1, b2, b3, b4))
print("BEST LIKELIHOOD: {0}".format(math.exp(best_likelihood)))


# Interact and test with new colors
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
