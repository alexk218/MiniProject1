import numpy as np

# Function to calculate entropy of a given sample set
def calculate_entropy(y):
    log2 = lambda x: np.log(x) / np.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

# Calculate entropy of a given split
def calculate_entropy_of_split(data, split_attribute, target_name):
    values, counts = np.unique(data[split_attribute], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) * calculate_entropy(
        data.where(data[split_attribute] == values[i]).dropna()[target_name]) for i in range(len(values))])
    return Weighted_Entropy

# Find the best attribute to split on
# Calculates the entropy before the split, the entropy of each split, and
# returns the attribute that results in the max. information gain (aka the maximum reduction in entropy)
def find_best_attribute(data, target_attribute_name):
    entropy_before_split = calculate_entropy(data[target_attribute_name])
    information_gains = []
    for attribute in data.columns:
        if attribute != target_attribute_name:
            entropy_of_split = calculate_entropy_of_split(data, attribute, target_attribute_name)
            information_gain = entropy_before_split - entropy_of_split
            information_gains.append((attribute, information_gain))
    best_attribute = max(information_gains, key=lambda x: x[1])
    return best_attribute[0]
