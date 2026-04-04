import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import tensorflow as tf
from models.moe_model import HierarchicalMoE

app = Flask(__name__, static_folder='static')
CORS(app)

print("Loading Hierarchical MoE Model...")
try:
    moe = HierarchicalMoE('saved_models')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    moe = None

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if moe is None:
        return jsonify({'error': 'MoE model failed to load on the server'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    try:
        # Load and preprocess image (CIFAR-10 is 32x32)
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((32, 32))
        img_array = np.array(image) / 255.0
        x = np.expand_dims(img_array, axis=0) # shape (1, 32, 32, 3)
        
        # Get final combined prediction
        final_preds = moe.predict(x, verbose=0)
        pred_label_idx = int(np.argmax(final_preds, axis=1)[0])
        pred_label = CIFAR10_CLASSES[pred_label_idx]
        confidence = float(final_preds[0][pred_label_idx])
        
        # --- Extract Dynamic Routing Information ---
        # Level 1 Routing (Art vs Nat)
        p_nat_array = moe.first_level_gate.predict(x, verbose=0)
        p_nat = float(p_nat_array[0, 0])
        p_art = 1.0 - p_nat
        
        # Level 2 Routing (Base vs Spec for Artificial)
        art_weights = moe.art_gater.predict(x, verbose=0)
        w_art_base = float(art_weights[0, 0])
        w_art_spec = float(art_weights[0, 1])
        
        # Level 2 Routing (Base vs Spec for Natural)
        nat_weights = moe.nat_gater.predict(x, verbose=0)
        w_nat_base = float(nat_weights[0, 0])
        w_nat_spec = float(nat_weights[0, 1])
        
        response = {
            'prediction': pred_label,
            'confidence': confidence,
            'routing': {
                'p_natural': p_nat,
                'p_artificial': p_art,
                'art_weights': {'base': w_art_base, 'specialized': w_art_spec},
                'nat_weights': {'base': w_nat_base, 'specialized': w_nat_spec}
            }
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask Server. Visit http://localhost:5000 in your browser.")
    app.run(debug=True, port=5000)
