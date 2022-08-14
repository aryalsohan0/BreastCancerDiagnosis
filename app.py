from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)

model = pickle.load(open("final_model.pkl", "rb"))

@app.route("/", methods = ["GET"])

def Home():
    return render_template('index.html')


standard_to = scaler.transform()
@app.route("/predict", methods=['POST'])

def predict():
    if request.method == "POST":
        new_obs = np.array(request.form["new_obs"])
            
            
        scaled_new_obs = scaler.transform(new_obs)
        
        y_proba = rf_clf.predict_proba(X)[0,1]
        pred = y_proba > 0.5
        
        if pred == 1:
            return render_template('index.html',prediction_texts="Breast lump is predicted as benign.")
        else:
            return render_template('index.html',prediction_text="Breast lump is predicted as malignant.")
            
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)