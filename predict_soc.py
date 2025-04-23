import pandas as pd
import joblib

# Load the trained model
model = joblib.load('soc_model.pkl')

# Example input (new data point to predict SoC)
new_data = pd.DataFrame([[230, 6.5, 45]], columns=['Voltage', 'Current', 'Temperature'])

# Predict SoC (State of Charge)
predicted_soc = model.predict(new_data)
print("Predicted SoC:", predicted_soc[0])
