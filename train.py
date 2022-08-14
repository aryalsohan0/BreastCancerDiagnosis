#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle




output_file = "randomForestmodel.bin"


data = pd.read_csv("data.csv")



data = data.drop("Unnamed: 32", axis = 1)






# Preparing Data
print("preparing data for train")

X = data.iloc[:,2:].values
y = data.iloc[:,1].values





encoder = LabelEncoder()
y = encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def modelsearch(classifier, parameters, scoring, X_train, y_train):
    
    model = GridSearchCV(estimator=classifier,
                   param_grid=parameters,
                   scoring= scoring,
                   cv=10,
                   verbose=1,
                   n_jobs=-1)
    
    model.fit(X_train, y_train)
    cv_results = model.cv_results_
    best_parameters = model.best_params_
    best_result = model.best_score_
    print('The best parameters for classifier is', best_parameters)
    print('The best training score is %.3f:'% best_result)
    
    return cv_results, best_parameters, best_result



print("Fitting the Random Forest Model")


rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [20, 50, 100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

cv_results, best_param, best_result = modelsearch(classifier = rf, parameters = param_grid, scoring = "f1",
                                                  X_train = X_train, y_train = y_train)




rf_clf = RandomForestClassifier(n_estimators = best_param['n_estimators'],
                                criterion = best_param['criterion'],
                                bootstrap = best_param['bootstrap'],
                                random_state=42)
rf_clf.fit(X_train, y_train)


rf_pred = rf_clf.predict(X_test)

print("\nClassification Report on Test Data\n")
print(classification_report(y_test, rf_pred))


with open('model.bin', 'wb') as f_out:
   pickle.dump((scaler, rf_clf), f_out)
f_out.close() ## After opening any file it's nessecery to close it



#pip list --format=freeze > requirements.txt
