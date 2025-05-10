from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Dummy model
from sklearn.linear_model import LogisticRegression
import os

model_path = "model.pkl"
if not os.path.exists(model_path):
    X = np.random.rand(100, 7)
    y = np.random.randint(0, 2, 100)
    model = LogisticRegression()
    model.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

with open(model_path, "rb") as f:
    model = pickle.load(f)

HTML_FORM = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heart Risk Calculator</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 20px; }
        input, button { margin: 5px; padding: 8px; width: 200px; }
        #result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Heart Risk Calculator</h1>
    <form id="riskForm">
        <input name="age" type="number" placeholder="Age" required><br>
        <input name="gender" type="number" placeholder="Gender (0=F,1=M)" required><br>
        <input name="sbp" type="number" placeholder="Systolic BP" required><br>
        <input name="chol" type="number" placeholder="Total Cholesterol" required><br>
        <input name="hdl" type="number" placeholder="HDL Cholesterol" required><br>
        <input name="smoke" type="number" placeholder="Smoking (0/1)" required><br>
        <input name="diabetes" type="number" placeholder="Diabetes (0/1)" required><br>
        <button type="submit">Calculate Risk</button>
    </form>
    <div id="result"></div>
    <script>
        document.getElementById('riskForm').addEventListener('submit', async function (e) {
            e.preventDefault();
            const form = e.target;
            const data = Object.fromEntries(new FormData(form).entries());
            for (let key in data) data[key] = Number(data[key]);

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            document.getElementById('result').innerHTML = `
                <p>Risk Level: ${result.risk}</p>
                <p>Tips: ${result.tips}</p>
            `;
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array([
        data["age"], data["gender"], data["sbp"], data["chol"],
        data["hdl"], data["smoke"], data["diabetes"]
    ]).reshape(1, -1)

    result = model.predict(input_data)[0]
    risk = "High" if result == 1 else "Low"

    tips = (
        "Maintain a balanced diet, avoid smoking, and monitor blood pressure regularly."
        if risk == "High"
        else "Keep up your healthy lifestyle! Regular checkups are still important."
    )

    return jsonify({"risk": risk, "tips": tips})

if __name__ == '__main__':
    app.run(debug=True)