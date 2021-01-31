import numpy as np
from PIL import Image

import cnn_model
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    body_list = ["筋肉型","標準型","痩せ型","肥満型"]
    # submitした画像が存在したら、画像データをモデル用に整形
    try :
        if request.files['image']:
            img_file = request.files['image']
            temp_img = Image.open(request.files['image'])
            temp_img = temp_img.convert("RGB") #色空間をRGBに
            #今回は、モデルの精度を上げるために(64,64)で画像を学習させています。
            temp_img = temp_img.resize((64,64))
            temp_img = np.asarray(temp_img)
            im_rows = 64
            im_cols = 64
            im_color = 3
            in_shape = (im_rows,im_cols,im_color)
            nb_classes = 4
            img_array = temp_img.reshape(-1,im_rows,im_cols,im_color)
            img_array = img_array / 255
            model = cnn_model.get_model(in_shape,nb_classes)
            #学習済みモデルを呼び出す
            model.load_weights("photos-newmodel-light.hdf5")
            predict = model.predict([img_array])[0]
            index = predict.argmax()
            body_shape = body_list[index]
            body_score = 90*predict[0]+70*predict[1]+40*predict[2]+40*predict[3]
            body_score = int(body_score)

            return render_template('./result.html', title='結果',                  body_score=body_score,body_shape=body_shape,img_file=img_file)
    
    except:
        return render_template('./flask_api_index.html')
    
if __name__ == '__main__':
    
    app.debug = True
    app.run(host='localhost', port=8881)
