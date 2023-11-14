import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template


# Create flask app
flask_app = Flask(__name__,template_folder="template")
model=pickle.load(open("Heart_Attack.p","rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")

gender={'M':1,'F':0}
general={'N':0 ,'Y':1}
ChestPainType={'ATA':1 ,'NAP':2 ,'ASY':3 ,'TA':4}
RestingECG={'Normal':1, 'ST':2 ,'LVH':3}
ST_Slope={'Up':1 ,'Flat':2 ,'Down':3}
Heart={0:"Low risk of Heart Attack",1:"High risk of Heart Attack"}
@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features =[]
    float_features.append(int(request.form['Age']))
    float_features.append(gender[request.form['gender']])
    float_features.append(ChestPainType[request.form['ChestPainType']])
    float_features.append(float(request.form['RestingBP']))
    float_features.append(float(request.form['Cholesterol']))
    float_features.append(float(request.form['FastingBS']))
    float_features.append(RestingECG[request.form['RestingECG']])
    float_features.append(float(request.form['MaxHR']))
    float_features.append(general[request.form['ExerciseAngina']])
    float_features.append(float(request.form['Oldpeak']))
    float_features.append(ST_Slope[request.form['ST_Slope']])
    float_features=np.array(float_features)
    float_features=float_features.reshape(1,-1)
    predict=model.predict(float_features)
    result=Heart[predict[0]]
    return render_template("index.html", prediction_text = "You have {}".format(result))

if __name__ == "__main__":
    flask_app.run(debug=True)