from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import io
import cv2
from PIL import Image

app = Flask(__name__)
CORS(app) 

# Load the trained model
model = tf.keras.models.load_model('GranadeMDV01.keras')

# Define class names (ensure these match the ones used in training)
class_names = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']

def preprocessing(image):
    try:
        # Convert FileStorage object to BytesIO
        image_bytes = io.BytesIO(image.read())
        
        # Load image with PIL and preprocess
        img = Image.open(image_bytes)
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize((128, 128))  # Resize to model input size
        
        # Convert to numpy array and preprocess
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None



@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'fileup' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        image = request.files.get('fileup')
        if image.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400
        
        print(f"Received image: {image.filename}")
        
        image_arr = preprocessing(image)
        if image_arr is None:
            return jsonify({"error": "Error in preprocessing the image"}), 500
        
        print(f"Image array shape: {image_arr.shape}")
        
        result = model.predict(image_arr)
        ind = np.argmax(result)
        prediction = class_names[ind]
        prediction_confidence = float(result[0][ind])

        print(f"Prediction: {prediction}, Confidence: {prediction_confidence}")

        # return jsonify({"prediction": prediction, "confidence": prediction_confidence})
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
