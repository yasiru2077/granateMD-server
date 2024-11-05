// Required dependencies
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const path = require('path');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// Enable CORS
app.use(cors());

// Class names (ensure these match the ones used in training)
const class_names = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy'];

// Load model function
async function loadModel() {
    try {
        const model = await tf.loadLayersModel('file://GranadeMDV01/model.json');
        return model;
    } catch (error) {
        console.error('Error loading model:', error);
        throw error;
    }
}

// Image preprocessing function
async function preprocessing(buffer) {
    try {
        // Resize image to 128x128 and ensure it's RGB
        const processedImage = await sharp(buffer)
            .resize(128, 128)
            .ensureAlpha()
            .raw()
            .toBuffer();

        // Convert to tensor
        const tensor = tf.tensor4d(
            new Float32Array(processedImage),
            [1, 128, 128, 3]
        );

        return tensor;
    } catch (error) {
        console.error('Error in preprocessing:', error);
        return null;
    }
}

// Initialize model
let model;
(async () => {
    try {
        model = await loadModel();
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Failed to load model:', error);
    }
})();

// Prediction endpoint
app.post('/predict', upload.single('fileup'), async (req, res) => {
    try {
        // Check if file exists in request
        if (!req.file) {
            return res.status(400).json({ error: 'No file part in the request' });
        }

        console.log(`Received image: ${req.file.originalname}`);

        // Preprocess image
        const tensorImage = await preprocessing(req.file.buffer);
        if (!tensorImage) {
            return res.status(500).json({ error: 'Error in preprocessing the image' });
        }

        console.log(`Image tensor shape: ${tensorImage.shape}`);

        // Make prediction
        const predictions = await model.predict(tensorImage).array();
        const predictionArray = predictions[0];
        const maxIndex = predictionArray.indexOf(Math.max(...predictionArray));
        const prediction = class_names[maxIndex];
        const confidence = predictionArray[maxIndex];

        console.log(`Prediction: ${prediction}, Confidence: ${confidence}`);

        // Clean up tensor
        tensorImage.dispose();

        return res.json({ prediction });

    } catch (error) {
        console.error('Error in prediction:', error);
        return res.status(500).json({ error: error.message });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});