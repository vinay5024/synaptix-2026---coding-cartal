import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ===========================================================
# 1️⃣new LOAD DATASET
# ===========================================================
df = pd.read_csv("ai_match_dataset_final.csv")

# ===========================================================
# 2️⃣ FEATURE ENGINEERING
# ===========================================================
def count_items(value):
    if pd.isna(value) or value == "":
        return 0
    return len(value.split(","))

def get_skill_count(x):
    try:
        return len(json.loads(x))
    except:
        return 0

df["skill_count"] = df["skills_rating"].apply(get_skill_count)
df["required_skill_count"] = df["project_required_skills"].apply(count_items)
df["optional_skill_count"] = df["project_optional_skills"].apply(count_items)

df["difficulty_num"] = df["project_difficulty"].map({"Easy":1, "Medium":2, "Hard":3})
df["duration_num"] = df["project_duration"].str.extract("(\d+)").astype(int)

df["applicant_type_num"] = df["applicant_type"].map({
    "student":1, "fresher":2, "experienced":3
})

df["education_num"] = df["education"].map({
    "Diploma":1, "BSc":2, "B.Tech":3, "MSc":4, "M.Tech":5
})

# ===========================================================
# 3️⃣ SELECT CLEAN FEATURES
# ===========================================================
X = df[[
    "skill_count",
    "experience_years",
    "profile_completeness",
    "required_skill_count",
    "optional_skill_count",
    "difficulty_num",
    "duration_num",
    "applicant_type_num",
    "education_num"
]]

y = df["selection_status"]

# ===========================================================
# 4️⃣ TRAIN MODEL
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("\nModel trained successfully!")
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

# ===========================================================
# 5️⃣ SAVE MODEL USING JOBLIB
# ===========================================================
joblib.dump(model, "model.pkl")
print("\nmodel.pkl saved successfully!")

# ===========================================================
# 6️⃣ USER INPUTS
# ===========================================================
print("\n===============================")
print("     ENTER DETAILS BELOW       ")
print("===============================\n")

user_data = {}

print("Enter your skill count:")
print("(Sample: 5 — You know 5 skills like Python, Java, SQL...)")
user_data["skill_count"] = int(input("→ "))

print("\nEnter your experience (years):")
print("(Sample: 2 — 2 years internship/work)")
user_data["experience_years"] = float(input("→ "))

print("\nEnter your profile completeness (0-100):")
print("(Sample: 85 — Profile mostly filled)")
user_data["profile_completeness"] = float(input("→ "))

print("\nEnter required skills for project (comma separated):")
print("Sample: python,sql,flask")
required = input("→ ")
user_data["required_skill_count"] = len([s for s in required.split(",") if s.strip() != ""])

print("\nEnter optional skills (comma separated):")
print("Sample: docker,react")
optional = input("→ ")
user_data["optional_skill_count"] = len([s for s in optional.split(",") if s.strip() != ""])

print("\nDifficulty Level: (1=Easy, 2=Medium, 3=Hard)")
user_data["difficulty_num"] = int(input("→ "))

print("\nEnter project duration in months:")
user_data["duration_num"] = int(input("→ "))

print("\nApplicant Type (1=Student, 2=Fresher, 3=Experienced):")
user_data["applicant_type_num"] = int(input("→ "))

print("\nEducation (1=Diploma, 2=BSc, 3=B.Tech, 4=MSc, 5=M.Tech):")
user_data["education_num"] = int(input("→ "))

input_df = pd.DataFrame([user_data])

# ===========================================================
# 7️⃣ PREDICT
# ===========================================================
prediction = model.predict(input_df)[0]

print("\n===============================")
print("         RESULT                ")
print("===============================\n")

if prediction == 1:
    print("🎉 You are SUITABLE → SELECTED (1)")
else:
    print("❌ Not Suitable → NOT SELECTED (0)")