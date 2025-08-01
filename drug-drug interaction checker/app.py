from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


app = Flask(__name__)
CORS(app)

# Add CUDA device check
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Defining DrugInteractionModel class
class DrugInteractionModel(nn.Module):
    def __init__(self, input_size):
        super(DrugInteractionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize model and CUDA
model = DrugInteractionModel(input_size=14).to(device)
model.load_state_dict(torch.load('drug_interaction_model_best.pth', map_location=device))
model.eval()

# Helper functions
def encode_cyp3a4(cyp_value):
    encoding = np.zeros(4)
    encoding[int(cyp_value)] = 1
    return encoding

def predict_interaction(drug1_cyp3a4, drug1_binding_affinity, drug1_renal_clearance, drug1_logP, 
                       drug2_cyp3a4, drug2_binding_affinity, drug2_renal_clearance, drug2_logP):
    drug1_cyp = encode_cyp3a4(drug1_cyp3a4)
    drug2_cyp = encode_cyp3a4(drug2_cyp3a4)
    
    input_features = np.array([[drug1_binding_affinity, drug1_renal_clearance, drug1_logP,
                               drug2_binding_affinity, drug2_renal_clearance, drug2_logP] + 
                               drug1_cyp.tolist() + drug2_cyp.tolist()])
    
    input_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor).item()
        interaction = int(prediction > 0.5)
        return interaction, prediction

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        drug1_data = data['drug1']
        drug2_data = data['drug2']
        
        interaction, probability = predict_interaction(
            float(drug1_data['cyp3a4']),
            float(drug1_data['bindingAffinity']),
            float(drug1_data['renalClearance']),
            float(drug1_data['logp']),
            float(drug2_data['cyp3a4']),
            float(drug2_data['bindingAffinity']),
            float(drug2_data['renalClearance']),
            float(drug2_data['logp'])
        )
        
        return jsonify({
            'interaction': interaction,
            'probability': probability,
            'message': 'High probability of interaction' if interaction == 1 else 'Low probability of interaction'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/library')
def library():
    return render_template('library.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
