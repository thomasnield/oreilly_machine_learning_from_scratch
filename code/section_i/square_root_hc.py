import numpy as np

# Use hill climbing to estimate a square root

value = 11.0
sqrt_candidate = value / 2.0

for i in range(100000):
    adjust = np.random.standard_t(3, 1)[0]
    new_candidate =  sqrt_candidate + adjust 
    
    if  (value - sqrt_candidate ** 2) ** 2 > (value - new_candidate ** 2) ** 2:
        sqrt_candidate = new_candidate
        

print(sqrt_candidate)
