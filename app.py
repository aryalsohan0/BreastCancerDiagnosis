from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle




app = Flask(__name__)

with open('model.bin', 'rb') as f_in:
    scaler, rf_clf = pickle.load(f_in)
f_in.close()
    
@app.route("/", methods = ["GET"])

def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])

def predict():
    if request.method == "POST":
        #new_obs = np.array(request.form["new_obs"].replace("'","")).reshape(1,-1)
        new_obs_array = np.array( [float(i) for i in request.form["new_obs"].split(',')]).reshape(1,-1)
            
            
        scaled_new_obs = scaler.transform(new_obs_array)
        
        y_proba = rf_clf.predict_proba(scaled_new_obs)[0,1]
        
        
        if y_proba > 0.5:
            return render_template('index.html',prediction_text="Breast lump is predicted as malignant.")
        else:
            return render_template('index.html',prediction_text="Breast lump is predicted as benign.")
        
        
            
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)