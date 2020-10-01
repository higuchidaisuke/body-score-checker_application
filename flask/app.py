from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
from datetime import datetime
import os
import string
from PIL import Image
import cnn_model
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from flask import Flask, request
import numpy as np


app = Flask(__name__)
'''
def load_model():
    model = cnn_model.get_model(in_shape,nb_classes)
    model.load_weights("./image/photos-model-light_new.hdf5")

    print(' * Loading end')
'''

@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    # submitした画像が存在したら処理する
    if request.files['image']:
        img = Image.open(request.files['image'])
        img = img.convert("RGB") #色空間をRGBに
        img = img.resize((32,32)) #サイズ変更
        x = np.asarray(img)
        im_rows = 32
        im_cols = 32
        im_color = 3
        in_shape = (im_rows,im_cols,im_color)
        nb_classes = 4
        x = x.reshape(-1,im_rows,im_cols,im_color)
        x = x / 255
        model = cnn_model.get_model(in_shape,nb_classes)
        model.load_weights("photos-model-light_new_copy.hdf5")
        pre = model.predict([x])[0]
        body_score = 90*pre[0]+70*pre[1]+40*pre[2]+40*pre[3]
        '''
        # 類似度を出力
        label, predict_Confidence = recognizer.predict(image)
        predict_Confidence = str(predict_Confidence)
        # render_template('./result.html')
        '''
        return render_template('./result.html', title='結果', body_score=body_score)

if __name__ == '__main__':
    #load_model()
    app.debug = True
    app.run(host='localhost', port=8889)
