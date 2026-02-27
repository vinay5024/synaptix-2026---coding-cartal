import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ===============================================================
# 1️⃣ Load Dataset
# ===============================================================
df = pd.read_csv("ai_match_dataset_final.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# ===============================================================
# 2️⃣ Feature Engineering (Skill Count from JSON field)
# ===============================================================
def skill_count(x):
    try:
        d = json.loads(x)
        return len(d)
    except:
        return 0

df["skill_count"] = df["skills_rating"].apply(skill_count)

# ===============================================================
# 3️⃣ Drop Non-ML-Friendly Columns
# (text columns you won't use for baseline models)
# ===============================================================
drop_cols = [
    "name", "email", "skills", "skills_rating", "projects",
    "project_descriptions", "project_tech_stack", "github_link",
    "past_experience_desc", "hobbies", "resume_text",
    "project_description"
]

df = df.drop(columns=drop_cols)

# ===============================================================
# 4️⃣ Encode Categorical Columns
# ===============================================================
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col].astype(str))

# ===============================================================
# 5️⃣ Split Features & Target
# ===============================================================
X = df.drop("selection_status", axis=1)
y = df["selection_status"]

# ===============================================================
# 6️⃣ Train-Test Split
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================================
# 7️⃣ Scale Numeric Data (For Logistic, KNN, SVM)
# ===============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================================================
# 8️⃣ Prepare Model Dictionary
# ===============================================================
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

results = {}

# ===============================================================
# 9️⃣ Train and Evaluate All Models
# ===============================================================
for name, model in models.items():
    print(f"\nTraining: {name}")

    # scaled models
    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # accuracy and metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

# ===============================================================
# 🔟 Print Final Results
# ===============================================================
print("\n================ FINAL RESULTS ================")
for algo, metrics in results.items():
    print(f"\n{algo}:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")

print("\n===============================================")