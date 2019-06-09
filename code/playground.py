


feature_values = [0,2,4,6,8]
feature_values2 = feature_values.copy()
feature_values2.pop(0)

zip_avgs = [(zipped[0] + zipped[1]) / 2 for zipped in zip(feature_values, feature_values2)]

print(zip_avgs)