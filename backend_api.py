from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os
import shutil
import uuid
from pdf2image import convert_from_path
import mimetypes

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Define model loading
def load_model():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    num_classes = 4  # LL, LR, RL, RR
    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=num_classes)
    state_dict = torch.load('./EfficientNet/best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Image transformation pipeline
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.449], std=[0.226]),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

# Load model
model = load_model()
device = torch.device('cpu')
class_names = ['LL', 'LR', 'RL', 'RR']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create temporary folder
        temp_folder = os.path.join('temp', str(uuid.uuid4()))
        os.makedirs(temp_folder, exist_ok=True)

        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if not file:
            return jsonify({'error': 'Empty file uploaded'}), 400

        filename = file.filename
        temp_path = os.path.join(temp_folder, filename)
        file.save(temp_path)

        # Guess the file type
        mime_type, _ = mimetypes.guess_type(temp_path)
        print(f"File: {filename}, MIME type: {mime_type}")

        # Handle PDF files
        if mime_type == 'application/pdf':
            try:
                images = convert_from_path(temp_path)
                if not images:
                    raise Exception(f"No pages found in PDF: {filename}")
                temp_image_path = os.path.join(temp_folder, f"{filename}.jpg")
                images[0].save(temp_image_path, 'JPEG')
                temp_path = temp_image_path
            except Exception as e:
                raise Exception(f"Failed to convert PDF {filename}: {str(e)}")

        # Validate file
        try:
            img = Image.open(temp_path)
            img.verify()
            img.close()
            img = Image.open(temp_path)
        except UnidentifiedImageError as e:
            raise Exception(f"Invalid image file: {filename}. Ensure it's a valid JPEG, PNG, or PDF.")
        except Exception as e:
            raise Exception(f"Failed to process image {filename}: {str(e)}")

        # Transform and predict
        print(f"Processing image: {filename}")
        input_tensor = test_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)
            pred_class = class_names[pred_idx.item()]

        # Clean up temporary folder
        shutil.rmtree(temp_folder)

        return jsonify({'prediction': pred_class})
    except Exception as e:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        return jsonify({'error': str(e)}), 400