import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open(r'C:\Users\fofoh\Desktop\Automated Machine Learning for Breast Cancer Diagnosis Using TPOT\trained_model.pkl', 'rb'))

@app.route('/')
def home():
    # Renders the HTML form for input (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form submission
        data = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean'])
        ]

        # Convert the data to a DataFrame
        input_data = pd.DataFrame([data], columns=[
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'])

        # Make a prediction
        output = model.predict(input_data)[0]  # Get the prediction result

        # Map the output to a user-friendly format (assuming 1 is Malignant and 0 is Benign)
        result = 'Malignant' if output == 1 else 'Benign'

        # Render the result page with the prediction result
        return render_template("result.html", result=result)

    except KeyError as e:
        # If there is an issue with the form submission (e.g., missing data), display a basic error message
        return f"Missing or invalid input: {str(e)}", 400

    except Exception as e:
        # Generic error handling (e.g., prediction failure)
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
