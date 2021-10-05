from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = load_breast_cancer()
X = pd.DataFrame(df.data)
y = pd.DataFrame(df.target)

X.columns = df.feature_names
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_split = train_test_split(X, y, test_size=0.2)


log_clf = LogisticRegression()
forest_clf = RandomForestClassifier()
tree_clf = DecisionTreeClassifier() 

log_clf.fit(X_train, y_train)
forest_clf.fit(X_train, y_train)
tree_clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, f1_score, classification_report

ascore = accuracy_score(y_split, tree_p)
f1 = f1_score(y_split, tree_p)
clf_rep = classification_report(y_split, tree_p)

print('ascore', ascore)
print('f1', f1)
print('clf_rep', clf_rep)


