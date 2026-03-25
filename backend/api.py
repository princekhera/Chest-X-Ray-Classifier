from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("lung_xray_model_2.h5")

# Classes
classes = ["lung_opacity", "viral_pneumonia", "normal"]

app = Flask(__name__)
CORS(app)

# Preprocess function
def preprocess(image):
    image = image.resize((224, 224))  # resize to model input
    image = np.array(image) / 255.0    # normalize
    if image.shape[-1] == 4:           # remove alpha channel if present
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])

def predict():

    file = request.files["file"]

    image = Image.open(file.stream).convert("RGB")

    img = preprocess(image)

    prediction = model.predict(img)[0]

    result = {
        "lung_opacity": float(prediction[0]),
        "viral_pneumonia": float(prediction[1]),
        "normal": float(prediction[2])
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)