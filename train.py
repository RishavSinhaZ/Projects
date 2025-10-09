import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --------------------------
# 1️⃣ Load dataset
# --------------------------
df = pd.read_csv("Diversity_job_hiring_v2 - Sheet1.csv")

# --------------------------
# 2️⃣ Drop unnecessary / personal columns
# --------------------------
drop_cols = [
    "Timestamp", "Name", "Phone", "Email",
    "Caste/Community", "Has Disability", "Heard About Us Through"
]
df = df.drop(columns=drop_cols, errors="ignore")

# --------------------------
# 3️⃣ Clean data
# --------------------------
df = df.fillna("Unknown")   # replace missing with 'Unknown'

# Standardize inconsistent text
df = df.replace({
    "NA": "Unknown",
    "Not Sure": "Unknown",
    "Maybe / Prefer not to say": "Maybe"
})

# --------------------------
# 4️⃣ Define target and input features
# --------------------------
target = "Comfortable with AI Screening"

# Define features (X)
feature_cols = [
    "Gender", "Age Group", "Highest Education", "Employment Status",
    "Industry", "Company Size", "Job Level", "Work Experience (Years)",
    "Diversity Focus Area", "Rating"
]

X = df[feature_cols]
y = df[target]

# Encode target variable
y = y.replace({"No": 0, "Maybe": 1, "Yes": 2})

# --------------------------
# 5️⃣ Encode all categorical input columns
# --------------------------
le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col].astype(str))

# --------------------------
# 6️⃣ Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------
# 7️⃣ Train model
# --------------------------
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# --------------------------
# 8️⃣ Evaluate performance
# --------------------------
y_pred = model.predict(X_test)
print("\n✅ Model Training Complete!\n")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# 9️⃣ Save model for your team
# --------------------------
joblib.dump(model, 'ai_screening_model.pkl')
print("\n💾 Model saved as 'ai_screening_model.pkl'")
