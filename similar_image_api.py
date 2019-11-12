import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2
from flask import Flask, request, jsonify
from PIL import Image
import requests, os
from io import BytesIO
app = Flask(__name__)

if os.path.exists('product_id_list_deploy.txt'):
    PRODUCT_ID_LIST = list(np.loadtxt('product_id_list_deploy.txt', dtype=np.str))
else:
    PRODUCT_ID_LIST = []
if os.path.exists('features_deploy.npy'):
    FEATURES = np.load('features_deploy.npy')
else:
    FEATURES = []
MODEL = MobileNetV2(include_top=False, weights='imagenet', pooling="avg")
MODEL._make_predict_function()

def save_image_features(img, productID):
    img = np.expand_dims(img, axis=0)
    imgFeature = MODEL.predict(img)
    FEATURES.append(imgFeature[0])
    PRODUCT_ID_LIST.append(productID)
    np.save('features_deploy.npy'.format(suffix), features)
    np.savetxt('product_id_list_deploy.txt', PRODUCT_ID_LIST, fmt='%s')

def get_image_from_url(imgURL):
    img = requests.get(imgURL) # something like 'https://i.imgur.com/rqCqA.jpg'
    status_code = img.status_code
    if status_code != 200:
        return None, status_code
    img = Image.open(BytesIO(img.content))
    img = np.array(img)
    return img, status_code

@app.route("/<productID>/addImage", methods=['POST'])
def add_product(productID):
    req_data = request.get_json()
    imgURL = req_data['url']
    
    img, status_code = get_image_from_url(imgURL)

    if status_code != 200:
        message = {
            'message': 'Error in fetching image.'
        }
        resp = jsonify(message)
        resp.status_code = 400
        return resp

    save_image_features(img, productID)

    message = {
        'message': 'Image features added to the database.'
    }
    resp = jsonify(message)
    resp.status_code = 200
    return resp

@app.route("/<productID>/similarProducts", methods=['GET'])
def get_similar_product(productID):
    return productID

if __name__ == "__main__":
    app.run(host='0.0.0.0')