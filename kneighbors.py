import numpy as np
import pandas as pd
import operator
from collections import Counter 

def read_data(base_path, filename):
	csv_file = os.path.join(base_path, filename)
	df = pd.read_csv(csv_file)
	X = df.iloc[:, :-1]
	y = df.iloc[:, -1]
	return X, y

def fit(X, y):
	global X_global
	global y_global
	X_global = X
	y_global = y

def distance_between(x, X_test):
	return np.linalg.norm(x - X_test)

def predict(X_test, k):
	global X_global
	X = X_global
	global y_global
	y = y_global
	processed_observations = []
	for feature, label in zip(X, y):
		distance = distance_between(feature, X_test)
		processed_observations.append((label, distance))
	processed_observations = sorted(processed_observations, key=operator.itemgetter(1))[1:k+1]
	counter = Counter([item[0] for item in processed_observations]) 
	most_common_label = counter.most_common(1)[0][0]
	return most_common_label










