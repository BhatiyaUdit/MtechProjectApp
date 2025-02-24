import io

from PIL import Image
from flask import Flask, request, jsonify
import torch
import numpy as np
app = Flask(__name__)

model = torch.jit.load("./models/best.torchscript")
model.eval()  # Set the model to evaluation mode


# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    print("Size",img.size)
    img = img.resize((512, 512))  # Resize to the input size (adjust as needed)
    img = np.array(img) / 255.0  # Normalize the image to [0, 1]
    img = img.transpose(2, 0, 1)  # Change to CxHxW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return torch.tensor(img, dtype=torch.float32)  # Convert to a Torch tensor


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image file is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Preprocess the image
    input_image = preprocess_image(file.read())

    print("Input shape:", input_image.shape)

    # Run inference
    with torch.no_grad():  # Disable gradient calculation
        results = model(input_image)

    # Process results (this will depend on your specific model)
    # For demonstration purposes, we'll return the raw output as JSON
    # You may need to customize this based on your model's output format
    return jsonify(results.tolist())  # Convert the output tensor to list for JSON response

@app.route('/')
def home():
    return "Flask is running!dddd"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
