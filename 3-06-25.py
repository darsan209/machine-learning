from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and encoder
model = joblib.load("loan_model.pkl")

@app.route('/')
def home():
    return render_template("form.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        features = [
            int(request.form['Gender']),
            int(request.form['Married']),
            int(request.form['Dependents']),
            int(request.form['Education']),
            int(request.form['Self_Employed']),
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            int(request.form['Property_Area'])
        ]
        prediction = model.predict([np.array(features)])
        result = "Approved" if prediction[0] == 1 else "Rejected"
        return render_template("form.html", prediction_text=f"Loan Prediction: {result}")
    except:
        return render_template("form.html", prediction_text="Error in input. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
