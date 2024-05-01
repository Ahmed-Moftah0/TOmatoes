from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

app = Flask(__name__)


modelpath = "TomatoeFinal.h5"
MODEL = tf.keras.models.load_model(modelpath)

class_names = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']





ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/predict', methods=['POST']) #endpoint
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            image = file.read()
            image = np.array(Image.open(BytesIO(image)).resize((256, 256)))  # Pass size as a tuple
            image = np.expand_dims(image, 0)
            predictions = MODEL.predict(image)
            class_predicted = class_names[np.argmax(predictions)]
            confidence = round(100 * (np.max(predictions[0])), 2)
            return {'Prediction':class_predicted,'Confidence':confidence}
        except:
            return jsonify({'error': 'error during prediction'})

if __name__=='__main__':
    app.run()