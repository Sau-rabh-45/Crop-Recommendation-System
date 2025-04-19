from dataclasses import field

from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

app = Flask(__name__)

#Load model
with open("model/crop_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["Nitrogen"]),
        float(request.form["Phosporus"]),
        float(request.form["Potassium"]),
        float(request.form["Temperature"]),
        float(request.form["Humidity"]),
        float(request.form["Ph"]),
        float(request.form["Rainfall"]),
    ]
    prediction = model.predict([features])[0].title()
    return render_template("index.html", result = prediction)

if __name__ == "__main__":
    app.run(debug=True)
