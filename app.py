from flask import Flask, render_template, abort, jsonify, request
from google.protobuf import message
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from feature import *



app = Flask(__name__)


new_model = load_model('./my_model2')
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST','GET'])

def main():
    if request.method == "POST":
        query_title = request.form.get("title")
        query_text = request.form.get("maintext")
        query = query_title + " " + query_text
        print("PROCESSED:"+query)
        preprocessed_input = preprocessing(query)
        print("Afterrr:")
        print(preprocessed_input)

        # x=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129, 6086, 416, 5766, 756, 2762, 3858, 6436, 24, 18, 2696, 1811, 3, 4, 73, 123, 342, 653, 1321, 8130, 5, 255, 17, 508, 59, 33, 1130, 494, 1066, 6, 685, 264, 105, 105, 256, 11, 972, 131, 4750, 3858, 378, 109, 615, 17, 1050, 106, 1048, 2681, 105, 451, 256, 1412, 972, 131, 5837, 1598, 2762, 3858, 91, 228, 598, 4371, 4750, 4773, 8462, 3590, 9604, 1715, 2275, 7952, 3401, 2731, 12, 1674, 3024, 2719, 6518, 7192, 824, 22, 13, 301, 3351, 3638, 2920, 9436, 1773, 761, 3952, 2259, 885, 404, 105, 451, 256, 1676, 21, 9157, 106, 579, 86, 353, 141, 18, 354, 84, 415, 2762, 3858, 1168, 228, 1043, 44, 7, 8, 3638, 2, 1008, 62, 5103, 271, 2153, 945, 19, 76, 404, 105, 451, 256, 102, 105, 105, 256, 9277, 1317, 554, 99, 228, 201, 933, 2762, 3858, 758, 3, 4, 10, 2704, 849, 215, 5820, 8437, 9, 1025, 73, 6, 685, 17, 404, 105, 451, 256, 733, 1412, 972, 365, 9, 6, 2213, 61, 810, 733, 105, 105, 256, 28, 2036, 6, 1658, 626, 1171, 2893, 1064, 2762, 1261, 73, 1497, 2, 738, 9, 6, 86, 180, 536, 382, 105, 105, 256, 2762, 392, 972, 38, 73, 7074, 9, 2139, 4285, 5444, 13, 661, 59, 2133, 2399, 11, 2762, 609, 66, 1497, 28, 328, 19, 1497, 2, 1945, 13, 850, 2051, 203, 2762, 395, 1652, 32, 9442, 15, 1, 1388, 2821, 67, 357, 88, 8981, 1270, 1581, 891, 395, 418, 6940, 3758, 13, 416, 8021, 11, 1270, 1581, 62, 121, 3458, 738, 328, 13, 19, 1531, 1317, 1186, 395, 418, 1451, 761, 88, 8, 73, 4687, 1162, 625, 6086, 2139, 6644, 626, 62, 73, 884, 8442, 123, 73, 3907, 494, 6086, 2762, 609, 416]]
        predd = new_model.predict_classes(preprocessed_input)
        print(predd[0][0])
        if predd[0][0] == 0:
            return render_template('index.html', message="False News")
        else:
            return render_template('index.html', message="True News")

   

if __name__ == '__main__':
    app.run(port=2080, debug=True)