import sklearn.datasets
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.cross_validation
import numpy

df = sklearn.datasets.load_boston()

X = df.data
y = df.target

X_scaled = sklearn.preprocessing.scale(X)

kf = sklearn.cross_validation.KFold(n = len(y), n_folds = 5, shuffle = True, random_state = 42)
knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 5, weights = 'distance', metric = 'minkowski')

mean_squared_error = sklearn.cross_validation.cross_val_score(knn, X_scaled, y, cv = kf, scoring = 'mean_squared_error')
# print mean_squared_error

p_spaced = numpy.linspace(1.0, 10.0, num = 200)
# print p_spaced

scores = []
for p in p_spaced:
    knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 5, weights = 'distance', metric='minkowski', p = p)
    value = sklearn.cross_validation.cross_val_score(knn, X_scaled, y, cv = kf, scoring = 'mean_squared_error')
    scores.append([round(value.mean(), 5), p])
print scores
# print

print max(scores, key = lambda x: x[0])