"""
You have 100 patients. 45 complained of stomach pain after drinking beer,
the other 55 did not.

Of the 45 patients with stomach pain, 37 have Celiac disease.

Of the 55 patients without stomach pain, 7 have Celiac disease.

What is the weighted GINI impurity for using post-beer stomach pain
to determine whether a patient has Celiac disease?
"""

patient_count = 100

pain_count = 45
celiac_pain_count = 37

nonpain_count = 55
celiac_nonpain_count = 7

gini_impurity_pain = 1 - (celiac_pain_count / pain_count)**2  - \
                     ((pain_count - celiac_pain_count) / pain_count)**2


gini_impurity_nonpain = 1 - (celiac_nonpain_count / nonpain_count)**2  - \
                     ((nonpain_count - celiac_nonpain_count) / nonpain_count)**2

weighted_gini_impurity = (gini_impurity_pain * (pain_count / patient_count) +
                          gini_impurity_nonpain * (nonpain_count / patient_count)) 

print(weighted_gini_impurity)