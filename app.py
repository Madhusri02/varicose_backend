from flask import Flask, request, jsonify
from tensorflow import keras
from flask_cors import CORS 
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

MODEL_PATH = "varicose_vgg19_model.h5"
model = keras.models.load_model(MODEL_PATH)

def preprocess_image(image_file, img_size=(224, 224)):
    img = Image.open(image_file).convert('RGB')
    img = img.resize(img_size)
    img_arr = keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = tf.keras.applications.vgg19.preprocess_input(img_arr)
    pred = model.predict(img_arr)
    try:
        pred_value = np.asarray(pred).reshape(-1)[0]
    except Exception:
        pred_value = pred
    return float(pred_value)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image = request.files['image']
    try:
        prediction = preprocess_image(image)
    except Exception as e:
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500
    result = "varicose_veins" if prediction > 0.5 else "healthy"
    return jsonify({
        "probability": prediction,
        "result": result
    })

@app.route('/', methods=['GET'])
def home():
    return "Varicose Veins Detection API is running!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

