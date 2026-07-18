import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv("dataset.csv")

# Input features
X = data[["GDP", "Life_Expectancy", "Mean_Years_Schooling", "Expected_Years_Schooling"]]

# Output (Target)
y = data["HDI"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open("hdi_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")