from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import requests
import re
import os

# -----------------------------------------------------
# CREATE FLASK APP
# -----------------------------------------------------
app = Flask(__name__)
app.secret_key = "skilllens_secret_2025"

# -----------------------------------------------------
# DEBUG CHECKS
# -----------------------------------------------------
print("\n=== Flask App Starting ===")
print("Running from:", os.path.abspath(__file__))
print("Templates folder:", os.path.isdir("templates"))
print("index.html exists:", os.path.isfile("templates/index.html"))

# -----------------------------------------------------
# HOME → Renders index.html
# -----------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------------------------------
# FRONTEND ROUTES
# -----------------------------------------------------
@app.route("/index")
def index_page():
    return render_template("index.html")

@app.route("/auth")
def auth():
    return render_template("auth.html")

@app.route("/company")
def company_dashboard():
    return render_template("company_dashboard.html")

@app.route("/jobseeker")
def jobseeker_dashboard():
    return render_template("jobseeker_dashboard.html")

@app.route("/profile")
def profile_edit():
    return render_template("profile_edit.html")

# NEW — Skill DNA page
@app.route("/skilldna")
def skilldna():
    return render_template("skilldna_match.html")

# -----------------------------------------------------
# LOAD ML MODEL
# -----------------------------------------------------
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    print("Loading ML model...")
    model = joblib.load(MODEL_PATH)
    print("ML Model loaded successfully!")
else:
    print("WARNING: model.pkl NOT FOUND — ML disabled")
    model = None

# -----------------------------------------------------
# HELPER: GITHUB ANALYSIS
# -----------------------------------------------------
def analyze_github(repo_url):
    try:
        match = re.search(r"github\.com/(.+?)/(.+?)(?:$|/)", repo_url)
        if not match:
            return {"error": "Invalid GitHub URL"}

        username, repo = match.groups()

        repo_info = requests.get(
            f"https://api.github.com/repos/{username}/{repo}"
        ).json()

        lang_info = requests.get(
            f"https://api.github.com/repos/{username}/{repo}/languages"
        ).json()

        score = len(lang_info) * 10 + repo_info.get("size", 0)

        return {
            "languages": lang_info,
            "stars": repo_info.get("stargazers_count", 0),
            "complexity_score": score
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------------------------------
# API ROUTES
# -----------------------------------------------------
@app.route("/api/test")
def api_test():
    return jsonify({"message": "API is working"})

@app.route("/api/predict", methods=["POST"])
def predict_json():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    df_input = pd.DataFrame([data])

    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    return jsonify({
        "prediction": int(pred),
        "probability": float(prob)
    })

@app.route("/api/github", methods=["POST"])
def github_api():
    url = request.json.get("github_url", "")
    result = analyze_github(url)
    return jsonify(result)

# -----------------------------------------------------
# HTML PREDICTION ROUTE (FOR UI PAGE)
# -----------------------------------------------------
@app.route("/predict_skilldna", methods=["POST"])
def predict_skilldna():
    if model is None:
        return "Model not loaded"

    # Read form data
    input_data = request.form.to_dict()
    df_input = pd.DataFrame([input_data])

    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    return render_template(
        "skilldna_result.html",
        prediction=int(pred),
        probability=float(prob)
    )

# -----------------------------------------------------
# RUN SERVER
# -----------------------------------------------------
if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    app.run(port=5000, debug=False, use_reloader=False)