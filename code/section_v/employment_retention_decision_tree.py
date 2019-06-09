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
    def __init__(self, feature_name, feature_type, value_extractor):
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.value_extractor = value_extractor

    def __str__(self):
        return self.feature_name


features = [Feature("sex", "binary", lambda emp: emp.sex),
            Feature("age", "numeric", lambda emp: emp.age),
            Feature("Promotions", "numeric", lambda emp: emp.promotions),
            Feature("Years Employed", "numeric", lambda emp: emp.years_employed)]

outcomes = ["WILL_STAY", "WILL_LEAVE"]


def gini_impurity(sample_employees):
    quit_employees = [employee for employee in sample_employees if employee.did_quit]
    non_quit_employees = [employee for employee in sample_employees if not employee.did_quit]

    return 1.0 - (len(quit_employees) / (len(sample_employees) + .0001)) ** 2 - (
            len(non_quit_employees) / (len(sample_employees) + .0001)) ** 2


def gini_impurity_for_feature(feature, boundary_value, sample_employees):
    feature_positive_employees = [employee for employee in sample_employees if
                                  feature.value_extractor(employee) >= boundary_value]
    feature_negative_employees = [employee for employee in sample_employees if
                                  feature.value_extractor(employee) < boundary_value]

    return ((len(feature_positive_employees) / len(sample_employees)) * gini_impurity(feature_positive_employees)) + (
            (len(feature_negative_employees) / len(sample_employees)) * gini_impurity(feature_negative_employees))

def split_continuous_variable(feature, sample_employees):
    feature_values = list(set(feature.value_extractor(employee) for employee in sample_employees))
    feature_values2 = feature_values.copy()
    feature_values2.pop(0)

    best_impurity = 1.0
    best_split = None
    zipped_values = zip(feature_values, feature_values2)

    for pair in zipped_values:
        split_value = (pair[0] + pair[1]) / 2
        impurity = gini_impurity_for_feature(feature,split_value, sample_employees)
        if impurity < best_impurity:
            best_impurity = impurity
            best_split = split_value

    return best_split

class TreeLeaf:
    def __init__(self, feature, feature_value):
        self.feature = feature
        self.feature_value = feature_value


for feature in features:
    feature_avg = sum([ feature.value_extractor(employee) for employee in all_employees]) / len(all_employees)
    print("{0}: {1}".format(feature, gini_impurity_for_feature(feature, split_continuous_variable(feature,all_employees), all_employees)))
    print("{0}: {1}".format(feature, split_continuous_variable(feature,all_employees)))
