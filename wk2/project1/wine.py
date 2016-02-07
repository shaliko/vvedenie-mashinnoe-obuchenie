import pandas
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing


df = pandas.read_csv('wine.data', header=None)

X = df.ix[:,1:]
y = df[0]

kf = KFold(n=len(y), n_folds=5, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
# print scores
# print scores.mean()


scores = []
for neighbor in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors = neighbor)
    values = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    print neighbor, round(values.mean(), 5)
    scores.append(round(values.mean(), 5))

# print max(scores) #
# Index 1 value 0.73


scores = []
X_scaled = preprocessing.scale(X)
for neighbor in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors = neighbor)
    values = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
    print neighbor, round(values.mean(), 5)
    scores.append(round(values.mean(), 5))

print max(scores) #
# Index 29 value 0.977619