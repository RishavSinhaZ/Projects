import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("/var/lock/CSV/Diversity_job_hiring_v2 - Sheet1.csv")

# 2. Drop columns that are not relevant for prediction
drop_cols = [
    "Timestamp", "Name", "Phone", "Email",
    "Caste/Community", "Has Disability", "Heard About Us Through"
]
df = df.drop(columns=drop_cols, errors="ignore")

# 3. Basic cleaning and normalization
df = df.fillna("Unknown")
df = df.replace({
    "NA": "Unknown",
    "Not Sure": "Unknown",
    "Maybe / Prefer not to say": "Maybe"
})

# 4. Define target and feature columns
target = "Comfortable with AI Screening"
feature_cols = [
    "Gender", "Age Group", "Highest Education", "Employment Status",
    "Industry", "Company Size", "Job Level", "Work Experience (Years)",
    "Diversity Focus Area", "Rating"
]

X = df[feature_cols]
y = df[target]

# Encode target
y = y.replace({"No": 0, "Maybe": 1, "Yes": 2})

# 5. Encode categorical features
le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col].astype(str))

# 6. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Train the Random Forest model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate model performance
y_pred = model.predict(X_test)
print("\nModel Training Complete\n")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save the trained model
joblib.dump(model, "ai_screening_model.pkl")
print("\nModel saved as 'ai_screening_model.pkl'")
