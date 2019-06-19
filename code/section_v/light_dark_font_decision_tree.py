import pandas as pd




class LabeledColor:
    def __init__(self, red, green, blue, dark_font_ind):
        self.red = red
        self.green = green
        self.blue = blue
        self.dark_font_ind = dark_font_ind

    def __str__(self):
        return "{0},{1},{2}: {3}".format(self.red, self.green, self.blue, self.dark_font_ind)



training_colors = [(LabeledColor(row[0], row[1], row[2], row[3])) for index, row in
                   pd.read_csv("https://tinyurl.com/y2qmhfsr").iterrows()]

class Feature:
    def __init__(self, feature_name, value_extractor):
        self.feature_name = feature_name
        self.value_extractor = value_extractor

    def __str__(self):
        return self.feature_name


features = [Feature("Red", lambda c: c.red),
            Feature("Green", lambda c: c.green),
            Feature("Blue", lambda c: c.blue)]

# Feature("Promotion Rate", lambda emp: emp.promotions / emp.years_employed)

outcomes = ["LIGHT", "DARK"]


def gini_impurity(sample_colors):
    dark_colors = [color for color in sample_colors if color.dark_font_ind == 1]
    light_colors = [color for color in sample_colors if color.dark_font_ind == 0]

    return 1.0 - (len(dark_colors) / (len(sample_colors) + .0001)) ** 2 - (
            len(light_colors) / (len(sample_colors) + .0001)) ** 2


def gini_impurity_for_feature(feature, split_value, sample_colors):
    feature_positive_elements = [color for color in sample_colors if
                                  feature.value_extractor(color) >= split_value]
    feature_negative_elements = [color for color in sample_colors if
                                  feature.value_extractor(color) < split_value]

    return ((len(feature_positive_elements) / len(sample_colors)) * gini_impurity(feature_positive_elements)) + (
            (len(feature_negative_elements) / len(sample_colors)) * gini_impurity(feature_negative_elements))


def split_continuous_variable(feature, sample_colors):
    feature_values = list(set(feature.value_extractor(color) for color in sample_colors))
    feature_values.sort()

    feature_values2 = feature_values.copy()
    feature_values2.pop(0)

    best_impurity = 1.0
    best_split = None
    zipped_values = zip(feature_values, feature_values2)

    for pair in zipped_values:
        split_value = (pair[0] + pair[1]) / 2
        impurity = gini_impurity_for_feature(feature, split_value, sample_colors)
        if impurity < best_impurity:
            best_impurity = impurity
            best_split = split_value

    return best_split


class TreeLeaf:

    def __init__(self, feature, split_value, sample_colors, previous_leaf):
        self.feature = feature
        self.split_value = split_value
        self.sample_colors = sample_colors
        self.previous_leaf = previous_leaf
        self.positive_colors = [e for e in sample_colors if feature.value_extractor(e) >= split_value]
        self.negative_colors = [e for e in sample_colors if feature.value_extractor(e) < split_value]
        self.gini_impurity = gini_impurity_for_feature(feature, split_value, sample_colors)

        self.positive_leaf = build_leaf(self.positive_colors, self)
        self.negative_leaf = build_leaf(self.negative_colors, self)

    def predict(self, color):
        print(self)
        feature_value = self.feature.value_extractor(color)
        if feature_value >= self.split_value:
            if self.positive_leaf is None:
                return len(self.positive_colors) / len(self.sample_colors)
            else:
                return self.positive_leaf.predict(color)
        else:
            if self.negative_leaf is None:
                return len(self.positive_colors) / len(self.sample_colors)
            else:
                return self.negative_leaf.predict(color)

    def __str__(self):
        return "{0} split on {1}, {2}|{3}, Impurity: {4}".format(self.feature, self.split_value,
                                                                 len(self.positive_colors),
                                                                 len(self.negative_colors), self.gini_impurity)




def build_leaf(sample_colors, previous_leaf):
    best_impurity = 1.0
    best_split = None
    best_feature = None

    for feature in features:
        split_value = split_continuous_variable(feature, sample_colors)

        if split_value is None:
            continue

        impurity = gini_impurity_for_feature(feature, split_value, sample_colors)

        if best_impurity > impurity:
            best_impurity = impurity
            best_feature = feature
            best_split = split_value

    if previous_leaf is None or best_impurity < previous_leaf.gini_impurity:
        return TreeLeaf(best_feature, best_split, sample_colors, previous_leaf)
    else:
        return None


tree = build_leaf(training_colors, None)


def recurse_and_print_tree(leaf, depth=0):
    if leaf is not None:
        print(("\t" * depth) + "({0}) ".format(depth) + str(leaf))
        recurse_and_print_tree(leaf.negative_leaf, depth + 1)
        recurse_and_print_tree(leaf.positive_leaf, depth + 1)


recurse_and_print_tree(tree)


print(tree.predict(LabeledColor(0,0,0,1)))