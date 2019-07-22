'''
    File name: load_forecasting_web.py
    Author: Spiros Politis
    Date created: 10/01/2019
    Date last modified: 17/01/2019
    Python Version: 3.6
'''

import datetime

from sklearn.externals import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from flask import Flask
from flask import request
from flask import jsonify

# Load the frozen model
with open("load_forecasting_model_v010.joblib", "rb") as model_file:
    model = joblib.load(model_file)

# Define the frozen model features
model_features = [
    "day_of_month", 
    "day_of_week", 
    "hour", 
    "lights", 
    "press_mm_hg", 
    "windspeed", 
    "tdewpoint", 
    "rh_in_mean" 
]

# Define the Flask app
app = Flask(__name__)

'''
    An exception to be thrown when the model parameters are missing or are wrong.
'''
class InvalidUsageException(Exception):
    # HTTP "Unprocessable entity" status code
    status_code = 422

    def __init__(self, message, status_code = None, payload = None):
        Exception.__init__(self)
        self.message = message

        if status_code is not None:
            self.status_code = status_code

        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


'''
    Checks whether all required request parameters are present.

    :param request_args: the request arguments captured from the app.route
'''
def check_parameters(request_args):
    if set(request_args) != set(model_features):
        raise InvalidUsageException(message = "Missing or wrong required parameters", status_code = 400)


'''
    Convert the request parameters to a Pandas DataFrame.

    :param request_args: the request arguments captured from the app.route
'''
def request_params_to_X_df(request_args):
    # Define a dataframe that will be passed as X to the model
    X_df = pd.DataFrame(index = [0], columns = model_features)

    for request_arg in request_args:
        X_df.loc[0, request_arg] = np.float64(request_args.get(request_arg))

    return X_df

'''
    Flask error handler for invalid parameters.

    :param InvalidUsageException: exception class that should be handled by the handler.
'''
@app.errorhandler(InvalidUsageException)
def handle_invalid_usage_esception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code

    return response


'''
    Define the /forecast route. Allow method HTTP GET only.

    :returns: JSON
'''
@app.route("/forecast", methods = ["GET"])
def forecast():
    # Check supplied route parameters
    check_parameters(request.args)

    # Parse request params and get the X dataframe 
    X_df = request_params_to_X_df(request.args)

    # Get prediction from model
    y_pred = model.predict(X_df)

    # Return JSON response
    return jsonify(
        X = request.args,
        predicted_value = y_pred[0],
        model_label = "RandomForestRegressor",
        timestamp = datetime.datetime.now(),
        comments = "X is the input vector, y is the predicted value"
    )