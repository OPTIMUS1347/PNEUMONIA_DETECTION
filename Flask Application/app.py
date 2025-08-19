import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

# Check for model file existence before loading
model_path = 'Flask Application/vgg_unfrozen.h5'
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
    print("Please check the file path and make sure it exists.")
    exit(1)

# Create model architecture
base_model = VGG19(include_top=False, input_shape=(128, 128, 3), weights=None)
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)

# Load weights with error handling
try:
    model_03.load_weights(model_path)
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

app = Flask(__name__)

# Create upload folder if it doesn't exist
upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
    
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "Normal"
    elif classNo == 1:
        return "Pneumonia"

def is_valid_xray_image(img_path):
    """
    Check if the uploaded image is likely to be a chest X-ray image
    by analyzing its characteristics.
    
    Parameters:
    img_path (str): Path to the image file
    
    Returns:
    bool: True if likely an X-ray, False otherwise
    str: Error or validation message
    """
    try:
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            return False, "Could not read image file"
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check image size
        height, width = gray.shape
        if height < 200 or width < 200:
            return False, "Image resolution too small for a valid X-ray"
        
        # Check if image has reasonable contrast (X-rays are typically high contrast)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        contrast = max_val - min_val
        if contrast < 50:
            return False, "Image contrast too low for a valid X-ray"
        
        # Calculate histogram and check distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        
        # X-rays typically have peaks in both dark and light regions
        dark_region = hist_norm[0:50].sum()
        mid_region = hist_norm[50:200].sum()
        light_region = hist_norm[200:].sum()
        
        
        # Check for color presence - X-rays are grayscale
        # Calculate average saturation from HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat_mean = np.mean(hsv[:,:,1])
        if sat_mean > 30:  # If average saturation is high, it's likely a color photo
            return False, "Image appears to be a color photo, not an X-ray"
        
        # If passed all checks, likely an X-ray
        return True, "Valid X-ray image"
    
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def getResult(img_path):
    try:
        # First validate if it's an X-ray image
        is_xray, message = is_valid_xray_image(img_path)
        if not is_xray:
            print(f"Image validation failed: {message}")
            return -1  # Special code for invalid X-ray
        
        # Use cv2 to read the image
        image = cv2.imread(img_path)
        if image is None:
            return "Error: Could not read image"
        
        # Convert BGR to RGB (since VGG expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (128, 128))
        
        # Preprocess the image same way as during training
        image = preprocess_input(image)
        
        # Expand dimensions to match model input shape
        input_img = np.expand_dims(image, axis=0)
        
        # Make prediction
        result = model_03.predict(input_img)
        
        # Get class with highest probability
        pred_class = np.argmax(result, axis=1)[0]
        confidence = result[0][pred_class] * 100
        
        print(f"Prediction: Class {pred_class} ({get_className(pred_class)}) with {confidence:.2f}% confidence")
        print(f"Raw probabilities: {result[0]}")
        
        return pred_class
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0  # Default to "Normal" on error

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            if not f:
                return jsonify({"error": "No file uploaded"}), 400
            
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            
            value = getResult(file_path)
            
            # Check if image is a valid X-ray
            if value == -1:
                is_valid, message = is_valid_xray_image(file_path)
                return jsonify({
                    "error": "Invalid X-ray image",
                    "message": "Please upload a valid chest X-ray image. " + message
                }), 400
            
            result = get_className(value)
            
            # Add more details to the result
            probability = model_03.predict(np.expand_dims(preprocess_input(cv2.resize(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB), (128, 128))), axis=0))
            confidence = float(probability[0][value])
            
            # Return JSON response
            return jsonify({
                "prediction": result.upper(),
                "confidence": confidence
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run(debug=True)