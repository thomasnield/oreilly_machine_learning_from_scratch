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
    quit_employees = [employee for employee in sample_employees if employee.did_quit]
    non_quit_employees = [employee for employee in sample_employees if not employee.did_quit]

    return 1.0 - (len(quit_employees) / (len(sample_employees) + .0001)) ** 2 - (
            len(non_quit_employees) / (len(sample_employees) + .0001)) ** 2


def gini_impurity_for_feature(feature, split_value, sample_employees):
    feature_positive_employees = [employee for employee in sample_employees if
                                  feature.value_extractor(employee) >= split_value]
    feature_negative_employees = [employee for employee in sample_employees if
                                  feature.value_extractor(employee) < split_value]

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
        impurity = gini_impurity_for_feature(feature, split_value, sample_employees)
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
        self.positive_employees = [e for e in sample_employees if feature.value_extractor(e) >= split_value]
        self.negative_employees = [e for e in sample_employees if feature.value_extractor(e) < split_value]
        self.gini_impurity = gini_impurity_for_feature(feature, split_value, sample_employees)

        self.positive_leaf = build_leaf(self.positive_employees, self)
        self.negative_leaf = build_leaf(self.negative_employees, self)

    def predict(self, employee):
        print(self)
        feature_value = self.feature.value_extractor(employee)
        if feature_value >= self.split_value:
            if self.positive_leaf is None:
                return len(self.positive_employees) / len(self.sample_employees)
            else:
                return self.positive_leaf.predict(employee)
        else:
            if self.negative_leaf is None:
                return len(self.positive_employees) / len(self.sample_employees)
            else:
                return self.negative_leaf.predict(employee)

    def __str__(self):
        return "{0} split on {1}, {2}|{3}, Impurity: {4}".format(self.feature, self.split_value,
                                                                 len(self.positive_employees),
                                                                 len(self.negative_employees), self.gini_impurity)




def build_leaf(sample_employees, previous_leaf):
    best_impurity = 1.0
    best_split = None
    best_feature = None

    for feature in features:
        split_value = split_continuous_variable(feature, sample_employees)

        if split_value is None:
            continue

        impurity = gini_impurity_for_feature(feature, split_value, sample_employees)

        if best_impurity > impurity:
            best_impurity = impurity
            best_feature = feature
            best_split = split_value

    if previous_leaf is None or best_impurity < previous_leaf.gini_impurity:
        return TreeLeaf(best_feature, best_split, sample_employees, previous_leaf)
    else:
        return None


tree = build_leaf(all_employees, None)


def recurse_and_print_tree(leaf, depth=0):
    if leaf is not None:
        print(("\t" * depth) + "({0}) ".format(depth) + str(leaf))
        recurse_and_print_tree(leaf.negative_leaf, depth + 1)
        recurse_and_print_tree(leaf.positive_leaf, depth + 1)


recurse_and_print_tree(tree)


print(tree.predict(EmployeeRetention(0,44,3,10,0)))