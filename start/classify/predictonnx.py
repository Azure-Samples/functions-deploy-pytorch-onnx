import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
from urllib.request import urlopen
import json
import time

import logging
import os
import sys
from datetime import datetime

# display images in notebook
from PIL import Image, ImageDraw, ImageFont, ImageOps

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

# Run the model on the backend
d=os.path.dirname(os.path.abspath(__file__))
modelfile=os.path.join(d , 'model.onnx')
labelfile=os.path.join(d , 'labels.json')

session = onnxruntime.InferenceSession(modelfile, None)

# get the name of the first input of the model
input_name = session.get_inputs()[0].name  

labels = load_labels(labelfile)

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    #add batch channel
    norm_img_data = norm_img_data.reshape(1,3,224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

def predict_image_from_url(image_url):
    with urlopen(image_url) as testImage:
        image = Image.open(testImage)

    imnew=ImageOps.fit(image, (224,224))

    image_data = np.array(imnew).transpose(2, 0, 1)
    input_data = preprocess(image_data)

    start = time.time()
    raw_result = session.run([], {input_name: input_data})
    end = time.time()
    res = postprocess(raw_result)

    inference_time = np.round((end - start) * 1000, 2)
    idx = np.argmax(res)

    response = {
            'created': datetime.utcnow().isoformat(),
            'prediction': labels[idx],
            'latency': inference_time,
            'confidence': res[idx]
    }
    logging.info(f'returning {response}')
    return response

if __name__ == '__main__':
    print(predict_image_from_url(sys.argv[1]))
