"""
You have 100 patients. 44 have Celiac disease, 56 do not. 

What is the GINI impurity of Celiac to non-Celiac patients?  
"""

patient_count = 100
celiac_count = 44
nonceliac_count  = 56

gini_impurity = 1 - (celiac_count / patient_count) ** 2 - (nonceliac_count / patient_count) ** 2

print(gini_impurity)