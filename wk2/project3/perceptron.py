import numpy
import pandas
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

train_df = pandas.read_csv('perceptron-train.csv')
y_train = train_df[train_df.columns[0]]
X_train = train_df[train_df.columns[1:]]

test_df = pandas.read_csv('perceptron-test.csv')
y_test = test_df[test_df.columns[0]]
X_test = test_df[test_df.columns[1:]]

clf = linear_model.Perceptron(random_state = 241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)
print accuracy # 0.361809045226

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)
scaled_predictions = clf.predict(X_test_scaled)

scaled_accuracy = metrics.accuracy_score(y_test, scaled_predictions)

print scaled_accuracy - accuracy