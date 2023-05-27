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
df = pd.read_csv('WZUMdataset.csv')

X = df.iloc[:, 1:64]
X['hand'] = df['handedness.label']
handLabelEncoder = LabelEncoder()
X['hand'] = handLabelEncoder.fit_transform(X['hand'])

joblib.dump(handLabelEncoder, "hand_label_encoder.pkl")


y = df.letter

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=77, test_size=0.15)


parameters = {
'kernel': ['poly', 'rbf'], 
'degree': [2],
'coef0' : [0.01, 0.005, 0.00],
'gamma' : ['scale'],
'C' : [10000]
}

gridSVC = GridSearchCV(SVC(), parameters)

clfs = [ gridSVC
]

print('\n####### RESULTS #######')
for clf in clfs:
    clf.fit(X_train, y_train)
    print("\n The best estimator across ALL searched params:\n",clf.best_estimator_)
    print("\n The best score across ALL searched params:\n",clf.best_score_)
    print("\n The best parameters across ALL searched params:\n",clf.best_params_)



#found using gridSearch
SVCmodel = SVC(C=10000, coef0=0.00, degree=2, kernel='poly', gamma='scale')

SVCmodel.fit(X_train, y_train)
print("SVC score: ", SVCmodel.score(X_test, y_test))
joblib.dump(SVCmodel, "SVCmodel.pkl")


# MLPmodel = MLPClassifier(hidden_layer_sizes=(100,100), activation='tanh', solver='adam', alpha=0.001, learning_rate='constant',
#                          max_iter=5000, beta_1=0.8, beta_2=0.88)

# MLPmodel.fit(X_train, y_train)
# print("MLP score: ", MLPmodel.score(X_test, y_test))
