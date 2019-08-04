import random

import pandas as pd


class EmployeeRetention:
    def __init__(self, sex, age, promotions, years_employed, did_quit):
        self.sex = sex
        self.age = age
        self.promotions = promotions
        self.years_employed = years_employed
        self.did_quit = did_quit


all_employees = [(EmployeeRetention(row[0], row[1], row[2], row[3], row[4])) for index, row in
                 pd.read_csv("https://tinyurl.com/y6r7qjrp").iterrows()]


class Feature:
    def __init__(self, feature_name, value_extractor):
        self.feature_name = feature_name
        self.value_extractor = value_extractor

    def __str__(self):
        return self.feature_name


features = [Feature("Sex", lambda emp: emp.sex),
            Feature("Age", lambda emp: emp.age),
            Feature("Promotions", lambda emp: emp.promotions),
            Feature("Years Employed", lambda emp: emp.years_employed)]

# Feature("Promotion Rate", lambda emp: emp.promotions / emp.years_employed)

outcomes = ["WILL_STAY", "WILL_LEAVE"]


def gini_impurity(sample_employees):
    quit_employees_ct = sum(1 for employee in sample_employees if employee.did_quit == 1)
    non_quit_employees_ct = sum(1 for employee in sample_employees if employee.did_quit == 0)
    sample_employees_ct = len(sample_employees)

    return 1.0 - (quit_employees_ct / sample_employees_ct) ** 2 - (non_quit_employees_ct / sample_employees_ct) ** 2


def gini_impurity_for_split(feature, split_value, sample_employees):
    feature_positive_employees = [employee for employee in sample_employees if feature.value_extractor(employee) >= split_value]
    feature_negative_employees = [employee for employee in sample_employees if  feature.value_extractor(employee) < split_value]

    return ((len(feature_positive_employees) / len(sample_employees)) * gini_impurity(feature_positive_employees)) + (
            (len(feature_negative_employees) / len(sample_employees)) * gini_impurity(feature_negative_employees))


def split_continuous_variable(feature, sample_employees):
    feature_values = list(set(feature.value_extractor(employee) for employee in sample_employees))
    feature_values.sort()

    feature_values2 = feature_values.copy()
    feature_values2.pop(0)

    best_impurity = 1.0
    best_split = None
    zipped_values = zip(feature_values, feature_values2)

    for pair in zipped_values:
        split_value = (pair[0] + pair[1]) / 2
        impurity = gini_impurity_for_split(feature, split_value, sample_employees)
        if impurity < best_impurity:
            best_impurity = impurity
            best_split = split_value

    return best_split


class TreeLeaf:

    def __init__(self, feature, split_value, sample_employees, previous_leaf):
        self.feature = feature
        self.split_value = split_value
        self.sample_employees = sample_employees
        self.previous_leaf = previous_leaf
        self.feature_positive_employees = [e for e in sample_employees if feature.value_extractor(e) >= split_value]
        self.feature_negative_employees = [e for e in sample_employees if feature.value_extractor(e) < split_value]
        self.weighted_gini_impurity = gini_impurity_for_split(feature, split_value, sample_employees)

        self.feature_positive_leaf = build_leaf(self.feature_positive_employees, self)
        self.feature_negative_leaf = build_leaf(self.feature_negative_employees, self)

    def predict(self, employee):
        feature_value = self.feature.value_extractor(employee)
        if feature_value >= self.split_value:
            if self.feature_positive_leaf is None:
                return sum(1 for e in self.feature_positive_employees if e.did_quit == 1) / len(self.feature_positive_employees)
            else:
                return self.feature_positive_leaf.predict(employee)
        else:
            if self.feature_negative_leaf is None:
                return sum(1 for e in self.feature_negative_employees if e.did_quit == 1) / len(self.feature_negative_employees)
            else:
                return self.feature_negative_leaf.predict(employee)

    def __str__(self):
        return "{0} split on {1}, {3}|{2}, Impurity: {4}".format(self.feature, self.split_value,
                                                                 len(self.feature_positive_employees),
                                                                 len(self.feature_negative_employees), self.weighted_gini_impurity)


def build_leaf(sample_employees, previous_leaf = None, random_feature_count = None ):
    best_impurity = 1.0
    best_split = None
    best_feature = None

    if random_feature_count is not None:
        sample_features = random.sample(features, random_feature_count)
    else:
        sample_features = features

    for feature in sample_features:
        split_value = split_continuous_variable(feature, sample_employees)

        if split_value is None:
            continue

        # Keep track of best feature with lowest impurity
        impurity = gini_impurity_for_split(feature, split_value, sample_employees)

        if best_impurity > impurity:
            best_impurity = impurity
            best_feature = feature
            best_split = split_value

    # The gini impurity must be improved by the next best split, otherwise the branch ends here
    if previous_leaf is None or best_impurity < gini_impurity(sample_employees):
        return TreeLeaf(best_feature, best_split, sample_employees, previous_leaf)
    else:
        return None



random_forest = [build_leaf(sample_employees=random.sample(all_employees, int(len(all_employees) * (2/3))), random_feature_count= random.choice(range(2,3))) for i in range(1,300)]

# Interact and test with new employee data
def predict_employee_will_stay(sex, age, promotions, years_employed):

    emp = EmployeeRetention(sex, age, promotions, years_employed, 0)
    will_leave_vote = sum(1 if tree.predict(emp) >= .5 else 0 for tree in random_forest)
    probability_of_leaving = will_leave_vote / len(random_forest)

    if probability_of_leaving >= .5:
        return "WILL LEAVE, {0}% chance of leaving\r\n".format(round(probability_of_leaving * 100.0, 2))
    else:
        return "WILL STAY, {0}% chance of leaving\r\n".format(round(probability_of_leaving * 100.0, 2))


while True:
    n = input("\r\nPredict employee will stay or leave {sex},{age},{promotions},{years employed}: ")
    (sex, age, promotions, years_employed) = n.split(",")
    print(predict_employee_will_stay(int(sex), int(age), int(promotions), int(years_employed)))
