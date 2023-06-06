import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"C:\Users\MyVampire\Desktop\New folder (3)\water_potability.csv")

# Split the dataset into training and testing sets
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    ph = float(request.form['ph'])
    hardness = float(request.form['hardness'])
    solids = float(request.form['solids'])
    chloramines = float(request.form['chloramines'])
    sulfate = float(request.form['sulfate'])
    conductivity = float(request.form['conductivity'])
    organic_carbon = float(request.form['organic_carbon'])
    trihalomethanes = float(request.form['trihalomethanes'])
    turbidity = float(request.form['turbidity'])

    # Create a new row with user input
    new_data = pd.DataFrame({'ph': [ph], 'Hardness': [hardness], 'Solids': [solids],
                             'Chloramines': [chloramines], 'Sulfate': [sulfate],
                             'Conductivity': [conductivity], 'Organic_carbon': [organic_carbon],
                             'Trihalomethanes': [trihalomethanes], 'Turbidity': [turbidity]})

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df.drop('Potability', axis=1))

    # Scale the data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_scaled, df['Potability'])

    # Make prediction on the new data point
    new_data_scaled = scaler.transform(imputer.transform(new_data))
    prediction = model.predict(new_data_scaled)

    # Determine the prediction result
    if prediction == 1:
        result = "The water is potable."
    else:
        result = "The water is not potable."

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
