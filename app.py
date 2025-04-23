from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("soc_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        voltage = float(request.form['Voltage'])
        current = float(request.form['Current'])
        temp = float(request.form['Temperature'])

        input_data = np.array([[voltage, current, temp]])
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction_text=f"Predicted SoC: {prediction:.2f}%")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
