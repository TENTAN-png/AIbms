import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the CSV file
data = pd.read_csv('battery_data.csv')

# Features and target
X = data[['Voltage', 'Current', 'Temperature']]
y = data['SoC']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'soc_model.pkl')

# Print training data ranges
print("\nTraining Data Ranges:")
print("Voltage:", data['Voltage'].min(), "to", data['Voltage'].max())
print("Current:", data['Current'].min(), "to", data['Current'].max())
print("Temperature:", data['Temperature'].min(), "to", data['Temperature'].max())
print("SoC:", data['SoC'].min(), "to", data['SoC'].max())

print("Model trained and saved Successfully!")
