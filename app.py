from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("hdi_model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the HTML form
    gdp = float(request.form["gdp"])
    life = float(request.form["life"])
    mean = float(request.form["mean"])
    expected = float(request.form["expected"])

    # Predict HDI
    prediction = model.predict([[gdp, life, mean, expected]])
    hdi = prediction[0]

    # Find HDI category
    if hdi < 0.55:
        category = "Low Human Development"
    elif hdi < 0.70:
        category = "Medium Human Development"
    elif hdi < 0.80:
        category = "High Human Development"
    else:
        category = "Very High Human Development"

    # Show result on the webpage
    return render_template(
        "index.html",
        prediction_text=f"Predicted HDI: {hdi:.3f}",
        category=category
    )

# Run the application
if __name__ == "__main__":
    app.run(debug=True)