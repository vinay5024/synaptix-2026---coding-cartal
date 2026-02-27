from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import requests
import re
import os

app = Flask(__name__)

print("\n========== Starting Flask App ==========\n")

# ============================================================
# LOAD MODEL SAFELY
# ============================================================
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: model.pkl not found in folder!")
    print("Place model.pkl in same folder as app.py")
    exit()

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model load failed:", str(e))
    exit()


# ============================================================
# SKILL COUNT FUNCTION
# ============================================================
def count_skill_items(value):
    if not value or value.strip() == "":
        return 0
    return len([s.strip() for s in value.split(",") if s.strip() != ""])


# ============================================================
# GITHUB ANALYSIS FUNCTION
# ============================================================
def analyze_github_repo(repo_url):
    try:
        pattern = r"github\.com\/(.+?)\/(.+?)(?:$|\/)"
        match = re.search(pattern, repo_url)

        if not match:
            return {"error": "Invalid GitHub URL"}

        username, repo = match.groups()
        api_url = f"https://api.github.com/repos/{username}/{repo}"

        repo_info = requests.get(api_url).json()
        lang_url = api_url + "/languages"

        languages = requests.get(lang_url).json()
        file_count = repo_info.get("size", 0)

        complexity_score = len(languages) * 10 + file_count

        return {
            "languages": languages,
            "file_count": file_count,
            "complexity_score": complexity_score,
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 1️⃣ PREDICTION API
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        required_count = count_skill_items(data.get("required_skills", ""))
        optional_count = count_skill_items(data.get("optional_skills", ""))

        user_input = {
            "skill_count": data.get("skill_count", 0),
            "experience_years": data.get("experience_years", 0),
            "profile_completeness": data.get("profile_completeness", 0),
            "required_skill_count": required_count,
            "optional_skill_count": optional_count,
            "difficulty_num": data.get("difficulty_num", 1),
            "duration_num": data.get("duration_num", 1),
            "applicant_type_num": data.get("applicant_type_num", 1),
            "education_num": data.get("education_num", 3)
        }

        df_input = pd.DataFrame([user_input])
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "match_probability": round(float(probability), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ============================================================
# 2️⃣ GITHUB ANALYSIS API
# ============================================================
@app.route("/analyze_github", methods=["POST"])
def github_analysis():
    try:
        link = request.json.get("github_link")
        result = analyze_github_repo(link)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


# ============================================================
# 3️⃣ MATCH PROJECT (GITHUB + ML)
# ============================================================
@app.route("/match_project", methods=["POST"])
def match_project():
    try:
        data = request.json
        github_link = data.get("github_link", "")

        gh = analyze_github_repo(github_link)
        project_score = gh.get("complexity_score", 0)

        required_count = count_skill_items(data.get("required_skills", ""))
        optional_count = count_skill_items(data.get("optional_skills", ""))

        user_input = {
            "skill_count": data.get("skill_count", 0),
            "experience_years": data.get("experience_years", 0),
            "profile_completeness": data.get("profile_completeness", 0),
            "required_skill_count": required_count,
            "optional_skill_count": optional_count,
            "difficulty_num": data.get("difficulty_num", 1),
            "duration_num": data.get("duration_num", 1),
            "applicant_type_num": data.get("applicant_type_num", 1),
            "education_num": data.get("education_num", 3)
        }

        df_input = pd.DataFrame([user_input])
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        return jsonify({
            "candidate_prediction": int(prediction),
            "match_probability": round(float(probability), 3),
            "github_analysis": gh,
            "overall_fit_score": round(probability * 100 + project_score * 0.1, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ============================================================
# RUN FLASK SERVER
# ============================================================
if __name__ == "__main__":
    print("🚀 Flask server running at: http://127.0.0.1:5000")
    app.run(debug=True)