from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('breast_cancer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/result', methods=['GET'])
def result_page():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    try:
        features = [
            float(data['radius_mean']), float(data['texture_mean']), float(data['perimeter_mean']),
            float(data['area_mean']), float(data['smoothness_mean']), float(data['compactness_mean']),
            float(data['concavity_mean']), float(data['concave_points_mean']), float(data['symmetry_mean']),
            float(data['fractal_dimension_mean']), float(data['radius_se']), float(data['texture_se']),
            float(data['perimeter_se']), float(data['area_se']), float(data['smoothness_se']),
            float(data['compactness_se']), float(data['concavity_se']), float(data['concave_points_se']),
            float(data['symmetry_se']), float(data['fractal_dimension_se']), float(data['radius_worst']),
            float(data['texture_worst']), float(data['perimeter_worst']), float(data['area_worst']),
            float(data['smoothness_worst']), float(data['compactness_worst']), float(data['concavity_worst']),
            float(data['concave_points_worst']), float(data['symmetry_worst']), float(data['fractal_dimension_worst'])
        ]
        features = np.array(features).reshape(1, -1)
        
        print("Features:", features)  

        prediction = model.predict(features)[0]
        print("Prediction:", prediction)  

  
        diagnosis = "Malignant" if prediction == 1 else "Benign"

    except Exception as e:
        print("Error in prediction:", e)
        diagnosis = "Error in prediction"
        
    return render_template('result.html', 
                           radius_mean=data['radius_mean'],
                           texture_mean=data['texture_mean'],
                           perimeter_mean=data['perimeter_mean'],
                           area_mean=data['area_mean'],
                           smoothness_mean=data['smoothness_mean'],
                           compactness_mean=data['compactness_mean'],
                           concavity_mean=data['concavity_mean'],
                           concave_points_mean=data['concave_points_mean'],
                           symmetry_mean=data['symmetry_mean'],
                           fractal_dimension_mean=data['fractal_dimension_mean'],
                           radius_se=data['radius_se'],
                           texture_se=data['texture_se'],
                           perimeter_se=data['perimeter_se'],
                           area_se=data['area_se'],
                           smoothness_se=data['smoothness_se'],
                           compactness_se=data['compactness_se'],
                           concavity_se=data['concavity_se'],
                           concave_points_se=data['concave_points_se'],
                           symmetry_se=data['symmetry_se'],
                           fractal_dimension_se=data['fractal_dimension_se'],
                           radius_worst=data['radius_worst'],
                           texture_worst=data['texture_worst'],
                           perimeter_worst=data['perimeter_worst'],
                           area_worst=data['area_worst'],
                           smoothness_worst=data['smoothness_worst'],
                           compactness_worst=data['compactness_worst'],
                           concavity_worst=data['concavity_worst'],
                           concave_points_worst=data['concave_points_worst'],
                           symmetry_worst=data['symmetry_worst'],
                           fractal_dimension_worst=data['fractal_dimension_worst'],
                           diagnosis=diagnosis)

if __name__ == '__main__':
    app.run(debug=True)
