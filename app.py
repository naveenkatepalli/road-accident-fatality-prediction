from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import sys
import os
from flask import Flask
app = Flask(__name__)  # URL Routing — Home Page
from sklearn.linear_model import LogisticRegression
ss=StandardScaler()
lor = LogisticRegression(random_state=0)


def algo(df):
    x = df.iloc[:, 0:5].values
    y = df.iloc[:,5].values
    x = ss.fit_transform(x)
    lor.fit(x, y)


def predict(people, day_week, hour, drunk_drive,month):
    a = lor.predict(ss.transform([[people, day_week, hour, drunk_drive,month]]))
    return int(a)


df = pd.read_csv(r"C:\Users\Naveen\OneDrive\Desktop\road accident\road accident data.csv")
model=algo(df)
@app.route("/",methods=["GET"])
def indexPage():
	return open(r"C:\Users\Naveen\OneDrive\Desktop\road accident\index.html").read()

@app.route("/number/",methods=["POST"])
def wordSearch():
	data = request.json
	result = predict(data["people"],data["day_week"], data["hour"], data["drunk_drive"],data["month"])
	return jsonify(result)

# Main Function, Runs at http://0.0.0.0:8000
if __name__ == "__main__":
    app.run(port=3000, debug=True)
