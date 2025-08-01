from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import io

app = Flask(__name__)
CORS(app)

# Load model artifacts once
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.joblib')
features = joblib.load('features.joblib')

def preprocess_input(data, is_batch=False):
    """Preprocess individual or batch data"""
    df = pd.DataFrame(data) if not is_batch else data
    
    # Convert month to season
    def month_to_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4]: return 'Spring'
        elif month in [5, 6]: return 'Summer'
        else: return 'Monsoon'
    
    df['SEASON'] = df['report_month'].apply(month_to_season)
    
    # Derived features
    df['IS_CHILD'] = (df['age'] <= 12).astype(int)
    df['IS_ELDERLY'] = (df['age'] >= 65).astype(int)
    df['IS_MONSOON'] = df['SEASON'].apply(lambda x: 1 if x in ['Monsoon', 'Summer'] else 0)
    df['CORE_SYMPTOMS'] = df[['fever', 'chills', 'headache']].sum(axis=1)
    df['CLASSICAL_TRIAD'] = ((df['fever'] == 1) & (df['chills'] == 1) & (df['headache'] == 1)).astype(int)
    df['DIGESTIVE_SYMPTOMS'] = df[['nausea', 'loss_of_appetite']].sum(axis=1)
    
    # Season encoding
    season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Monsoon': 3}
    df['SEASON'] = df['SEASON'].map(season_mapping).fillna(0).astype(int)
    
    # Ensure all required columns exist
    df = df.reindex(columns=features, fill_value=0)
    
    # Scale features
    return scaler.transform(df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/individual', methods=['POST'])
def predict_individual():
    try:
        # Get form data
        form_data = request.json
        input_data = {
            'AGE': form_data['age'],
            'DISTRICT': form_data['district'],
            'TEHSIL': form_data['tehsil'],
            'REPORT_YEAR': form_data['report_year'],
            'SEASON': form_data['report_month'],  # Will be converted in preprocess
            'FEVER': form_data['fever'],
            'CHILLS': form_data['chills'],
            'HEADACHE': form_data['headache'],
            'NAUSEA': form_data['nausea'],
            'LOSS_OF_APPETITE': form_data['loss_of_appetite']
        }
        
        # Preprocess and predict
        processed_data = preprocess_input([input_data])
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/dataset', methods=['POST'])
def predict_dataset():
    try:
        # Get uploaded file
        file = request.files['dataset']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
            
        # Read and process CSV
        df = pd.read_csv(file)
        processed_data = preprocess_input(df, is_batch=True)
        
        # Make predictions
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        
        # Add predictions to dataframe
        df['PREDICTION'] = predictions
        df['PROBABILITY'] = probabilities
        
        return jsonify({
            'results': df.to_dict(orient='records'),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)