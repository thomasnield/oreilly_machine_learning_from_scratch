import pandas as pd


class EmployeeRetention:
    def __init__(self, sex, age, promotions, years_employed, did_quit):
        self.sex = sex
        self.age = age
        self.promotions = promotions
        self.years_employed = years_employed
        self.did_quit = did_quit


employee_data = [(EmployeeRetention(row[0], row[1], row[2], row[3], row[4])) for index, row in
                 pd.read_csv("https://tinyurl.com/y6r7qjrp").iterrows()]

class Feature:
    def __init__(self, feature_name, feature_type):
        self.feature_name = feature_name
        self.feature_type = feature_type

features = [Feature("sex", "binary"),
            Feature("age", "numeric"),
            Feature("Promotions", "numeric"),
            Feature("Years Employed", "numeric"),
            Feature("Did Quit", "binary")]


class TreeNode:
    def __init__(self, feature):
        self.feature = feature
