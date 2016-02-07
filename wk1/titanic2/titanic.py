import pandas
df = pandas.read_csv('titanic.csv')

df = df[['Survived', 'Pclass', 'Fare', 'Sex', 'Age']]

df_cleaned = df[~np.isnan(df['Age'])]

# df_cleaned['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df_cleaned['Sex'] = df_cleaned["Sex"].apply(lambda sex: 0 if sex == "male" else 1)

X = df_cleaned[['Pclass', 'Fare', 'Sex', 'Age']]
y = df_cleaned[['Survived']]

import numpy as np
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
clf.feature_importances_