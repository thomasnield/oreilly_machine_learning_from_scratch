import pandas as pd


class WeatherItem:
    def __init__(self, rain, lightning, cloudy, temperature, good_weather_ind = None):
        self.rain = rain
        self.lightning = lightning
        self.cloudy = cloudy
        self.temperature = temperature
        self.good_weather_ind = good_weather_ind


all_samples = [(WeatherItem(row[0], row[1], row[2], row[3], row[4])) for index, row in
               pd.read_csv("https://tinyurl.com/y6o42f7v").iterrows()]


class Feature:
    def __init__(self, feature_name, value_extractor):
        self.feature_name = feature_name
        self.value_extractor = value_extractor

    def __str__(self):
        return self.feature_name


features = [Feature("Rain", lambda wi: wi.rain),
            Feature("Lightning", lambda wi: wi.lightning),
            Feature("Cloudy", lambda wi: wi.cloudy),
            Feature("Temperature", lambda wi: wi.temperature)]

# get impurity for provided samples
def gini_impurity(samples):
    good_weather_item_ct = sum(1 for weather_item in samples if weather_item.good_weather_ind == 1)
    bad_weather_item_ct = sum(1 for weather_item in samples if weather_item.good_weather_ind == 0)
    sample_ct = len(samples)

    return 1.0 - (good_weather_item_ct / sample_ct) ** 2 - (bad_weather_item_ct / sample_ct) ** 2


# get weighted impurity for entire node
def gini_impurity_for_split(feature, split_value, samples):
    feature_positive_items = [weather_item for weather_item in samples if feature.value_extractor(weather_item) >= split_value]
    feature_negative_items = [weather_item for weather_item in samples if feature.value_extractor(weather_item) < split_value]

    return (gini_impurity(feature_positive_items) * (len(feature_positive_items) / len(samples)) ) + (
            gini_impurity(feature_negative_items) * (len(feature_negative_items) / len(samples)) )


def split_continuous_variable(feature, samples):
    feature_values = list(set(feature.value_extractor(weather_item) for weather_item in samples))
    feature_values.sort()

    feature_values2 = feature_values.copy()
    feature_values2.pop(0)

    best_impurity = 1.0
    best_split = None
    zipped_values = zip(feature_values, feature_values2)

    for pair in zipped_values:
        split_value = (pair[0] + pair[1]) / 2
        impurity = gini_impurity_for_split(feature, split_value, samples)
        if impurity < best_impurity:
            best_impurity = impurity
            best_split = split_value

    return best_split


class TreeLeaf:

    def __init__(self, feature, split_value, samples):
        self.feature = feature
        self.split_value = split_value
        self.samples = samples
        self.feature_positive_items = [e for e in samples if feature.value_extractor(e) >= split_value]
        self.feature_negative_items = [e for e in samples if feature.value_extractor(e) < split_value]
        self.weighted_gini_impurity = gini_impurity_for_split(feature, split_value, samples)

        self.feature_positive_leaf = build_leaf(self.feature_positive_items, self)
        self.feature_negative_leaf = build_leaf(self.feature_negative_items, self)

    def predict(self, weather_item):
        print(self)
        feature_value = self.feature.value_extractor(weather_item)
        if feature_value >= self.split_value:
            if self.feature_positive_leaf is None:
                return sum(1 for e in self.feature_positive_items if e.good_weather_ind == 1) / len(self.feature_positive_items)
            else:
                return self.feature_positive_leaf.predict(weather_item)
        else:
            if self.feature_negative_leaf is None:
                return sum(1 for e in self.feature_negative_items if e.good_weather_ind == 1) / len(self.feature_negative_items)
            else:
                return self.feature_negative_leaf.predict(weather_item)

    def __str__(self):
        return "{0} split on {1}, {3}|{2}, Impurity: {4}|{5}, Weighted Impurity: {6}".format(self.feature, self.split_value,
                                                                 len(self.feature_positive_items),
                                                                 len(self.feature_negative_items),
                                                                 gini_impurity(self.feature_positive_items),
                                                                 gini_impurity(self.feature_negative_items),
                                                                 self.weighted_gini_impurity)


def build_leaf(weather_items, previous_leaf):
    best_impurity = 1.0
    best_split = None
    best_feature = None

    # Find feature with lowest impurity
    for feature in features:
        split_value = split_continuous_variable(feature, weather_items)

        # If value cannot be split, skip
        if split_value is None:
            continue

        impurity = gini_impurity_for_split(feature, split_value, weather_items)

        # Keep track of best feature with lowest impurity
        if best_impurity > impurity:
            best_impurity = impurity
            best_feature = feature
            best_split = split_value

    # The gini impurity of the samples must be improved with the next best split, otherwise the branch ends here
    if previous_leaf is None or gini_impurity(weather_items) > best_impurity:
        return TreeLeaf(best_feature, best_split, weather_items)
    else:
        return None


tree = build_leaf(all_samples, None)


def recurse_and_print_tree(leaf, depth=0):
    if leaf is not None:
        print(("\t" * depth) + "({0}) ".format(depth) + str(leaf))
        recurse_and_print_tree(leaf.feature_negative_leaf, depth + 1)
        recurse_and_print_tree(leaf.feature_positive_leaf, depth + 1)


recurse_and_print_tree(tree)


# Interact and test with new data
def predict_weather_will_be_good(rain, lightning, cloudy, temperature):
    rate_of_good_weather = tree.predict(WeatherItem(rain, lightning, cloudy, temperature))
    if rate_of_good_weather >= .5:
        return "Weather is good, {0}% records in node are good\r\n".format(round(rate_of_good_weather * 100.0, 2))
    else:
        return "Weather is bad, {0}%  records in node  are bad\r\n".format(round(rate_of_good_weather * 100.0, 2))


while True:
    n = input("\r\nPredict if weather is good {rain},{lightning},{cloudy},{temperature}: ")
    (rain, lightning, cloudy, temperature) = n.split(",")
    print(predict_weather_will_be_good(int(rain), int(lightning), int(cloudy), int(temperature)))
