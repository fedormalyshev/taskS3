import os
import sys
import logging
import requests
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

#Load VALUE model
response1 = requests.get('https://www.dropbox.com/s/f3zomahc0ri9q4j/VALUEreg.pkl?dl=1')
with open('VALUE_model', 'wb') as file1:
    file1.write(response1.content)
    
VALUE_model = joblib.load('VALUE_model')

#Load Y3 model
response2 = requests.get('https://www.dropbox.com/s/uqoxgs21ihnm5xs/Y3reg.pkl?dl=1')
with open('Y3_model', 'wb') as file2:
    file2.write(response2.content)

Y3_model = joblib.load('Y3_model')

#Load X3 model
response3 = requests.get('https://www.dropbox.com/s/khjte2pvdryh6q9/X3reg.pkl?dl=1')
with open('X3_model', 'wb') as file3:
    file3.write(response3.content)

X3_model = joblib.load('X3_model')

@app.route("/predict")
def predict():
    
    csv_url = request.args.get('csv_url')
    one_row = pd.read_csv(csv_url)
    v = VALUE_model.predict(one_row)
    y = Y3_model.predict(one_row)
    x = X3_model.predict(one_row)
    s = "Predicted values:\nX3: " +  str(x) + "\nY3: " + str(y) + "\nVALUE: " +str(v)
    return(s)

if __name__ == "__main__":
    app.run()
