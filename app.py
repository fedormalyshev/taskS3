import os
import sys
import logging
#import pickle
import requests
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route("/predict")
def predict():
    csv_url = request.args.get('csv_url')
    
    #one_row = pd.read_csv("https://www.dropbox.com/s/39dak20mpjmss0q/1_row_test.csv?dl=1")
    #x = VALUE_model.predict(one_row)
    return(csv_url)
    #return(type(VALUE_model))
    #i = request.args.get('i')
    #print("\n", i)

if __name__ == "__main__":
    response = requests.get('https://www.dropbox.com/s/f3zomahc0ri9q4j/VALUEreg.pkl?dl=1')
    with open('VALUE_model', 'wb') as file1:
        file1.write(response.content)
    
    VALUE_model = joblib.load('VALUE_model')
    
    #port = int(os.environ.get("VCAP_APP_PORT", 5000))
    #app.run(host='0.0.0.0', port=port)
    app.run()
