# -*- coding: utf-8 -*-

import os
from flask import Flask, request, jsonify
from fastai.vision.all import load_learner, PILImage

app = Flask(__name__)

# Load the trained model
model_path = '../Facial Recognition/models/vgg.pkl'
learn = load_learner(model_path)

def predict_image(img_path):
    # Open the image file
    img = PILImage.create(img_path)
    
    # Get prediction
    pred, pred_idx, probs = learn.predict(img)
    
    # Return the result as a dictionary
    return {
        'prediction': str(pred),
        'confidence': float(probs[pred_idx])
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the file to a temporary location
        img_path = os.path.join('temp', file.filename)
        file.save(img_path)
        
        # Get prediction
        result = predict_image(img_path)
        
        # Clean up temporary file
        os.remove(img_path)
        
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    # Create a temporary directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
