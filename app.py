#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flask
import pandas as pd
import numpy as np
import pickle
import joblib

from flask import request,render_template, url_for
app = flask.Flask(__name__)
app.config["DEBUG"] = True

from flask_cors import CORS
CORS(app)

# main index page route
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods=['POST'])
def predict():
    model = joblib.load(open("infant_prediction_model.pkl", "rb"))
    if request.method == "POST":
        # Date_of_Journey
        age= request.form["v013"]
        region= request.form["v024"]
        place_of_residence= request.form["v025"]
        educational_level= request.form["v106"]
        antenatal_visit= request.form["v130"]
        water_facility= request.form["v213"]
        household_member= request.form["v409"]
        frequency_of_using_radio= request.form["v411"]
        duration_of_current_pregnancy= request.form["v412a"]
        entries_in_birth_history= request.form["v412c"]
        the_pregnancy_is_wanted= request.form["v414e"]
        terminating_pregnancy= request.form["v414n"]
        contraceptive_use= request.form["v414s"]
        Breast_feeding_ststus= request.form["v414v"]
        Body_Mass_index= request.form["v445"]
        Husband_educational_level= request.form["v457"]
        husband_occupation= request.form["b9"]
        respondents_occupation= request.form["b19"]
        place_of_delivery= request.form["m4"]
        wealth_index= request.form["v190"]
        
        
    final_arr = [age,region,place_of_residence,antenatal_visit,educational_level,water_facility,household_member,
                 frequency_of_using_radio,duration_of_current_pregnancy,entries_in_birth_history,the_pregnancy_is_wanted,
                 terminating_pregnancy,contraceptive_use,Breast_feeding_ststus,Body_Mass_index,Husband_educational_level,
                 husband_occupation,respondents_occupation,place_of_delivery,wealth_index] 
    data = np.array(final_arr)
    data = data.reshape(1, -1)
    prediction=model.predict(data)
    output=round(prediction[0],2)
    if(output==1):
        return render_template('index.html',pred=f'Child Alive  {output}.\n')
    else:
        return render_template('index.html',pred=f'Child Died {output}')


if __name__ == "__main__":
    app.run(debug=False)


# In[ ]:




