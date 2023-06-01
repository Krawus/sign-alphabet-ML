import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, OrdinalEncoder
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import seaborn as sns
import joblib


#read data
df = pd.read_csv('WZUM_dataset.csv')

X = df.iloc[:, 1:64]
X['hand'] = df['handedness.label']
handLabelEncoder = LabelEncoder()
X['hand'] = handLabelEncoder.fit_transform(X['hand'])

joblib.dump(handLabelEncoder, "hand_label_encoder.pkl")

y = df.letter
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7, test_size=0.1)

#found using gridSearch
SVCmodel = SVC(C=10000, coef0=0.00, degree=2, kernel='poly', gamma='scale')

SVCmodel.fit(X_train, y_train)
print("SVC score: ", SVCmodel.score(X_test, y_test))
joblib.dump(SVCmodel, "SVCmodel.pkl")