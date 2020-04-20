from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/demo_test/')
def extract_sentiment():
    return "Workings"
