## Load dataset
import scipy.io as sio
import numpy as np

print("Preparing for dataset...")
mnist = sio.loadmat("mnist/mnist-original.mat")
X, y = mnist["data"], mnist["label"]
X_train, X_test = X[:, :60000], X[:, 60000:]
y_train, y_test = y[:, :60000], y[:, 60000:]

X_train = np.transpose(X_train)
X_test = np.transpose(X_test)
y_train = np.transpose(y_train)[:, 0]
y_test = np.transpose(y_test)[:, 0]

# create a classifier to do classification
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()

# create grid search to find the best parameters within search field
from sklearn.model_selection import GridSearchCV
param_grid = [
    {"weights": ["uniform", "distance"]},
    {"n_neighbors": [6, 7, 8, 9, 10, 11, 12]}
]
grid_search = GridSearchCV(knc, param_grid, cv=5, scoring="neg_mean_squared_error")
print("Searching for best parameters...")
grid_search.fit(X_train, y_train)
weights = grid_search.best_param_["weights"]
n_neighbors = grid_search.best_param_["n_neighbors"]

print("Training model...")
best_knc= grid_search.best_estimator_
best_knc.fit(X_train, y_train)
y_pred = best_knc.predict(X_test)

print("Evaluating...")
# evaluate the precision of the model
from sklearn.metric import precision_score
print(precision_score(y_test, y_pred))