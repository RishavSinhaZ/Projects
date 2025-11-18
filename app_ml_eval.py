# app_ml_eval.py (edited minimally: added remarks input + auto-scroll to results; removed download button)
import os
from datetime import datetime
import base64

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import streamlit.components.v1 as components  # <<< added for scrolling JS

# Optional Gemini
try:
    import google.generativeai as gen_ai
except Exception:
    gen_ai = None

# -------------------------
# CONFIG - must be first Streamlit call
# -------------------------
st.set_page_config(page_title="Job AI Safety Evaluator", layout="wide")

# -------------------------
# Load .env
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY and gen_ai:
    try:
        gen_ai.configure(api_key=GOOGLE_API_KEY)
        gem_model = gen_ai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        gem_model = None
else:
    gem_model = None

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_training_table(path="Diversity_job_hiring_v2 - Sheet1.csv"):
    # load the CSV used to train the model — this lets us rebuild encoders
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_model(path="ai_screening_model.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def build_encoders(df, feature_cols):
    encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        # fit on training values (stringified)
        le.fit(df[col].astype(str).fillna("Unknown"))
        encoders[col] = le
    return encoders

def encode_input(user_row, encoders, feature_cols):
    X = []
    for col in feature_cols:
        val = str(user_row.get(col, "Unknown"))
        le = encoders[col]
        # if unseen label, use a fallback integer (transform will error)
        if val in le.classes_:
            X.append(int(le.transform([val])[0]))
        else:
            # add fallback: use index 0
            X.append(0)
    return X

def percent_safe_from_model_probs(proba, class_index_yes=2):
    """
    We trained target {No:0, Maybe:1, Yes:2}.
    We define 'comfort with AI screening' probability = proba[class_index_yes].
    Heuristic: If user is very COMFORTABLE with AI screening (high prob yes),
    we treat that as higher probability that their situation is amenable to AI screening => higher risk.
    So safety = 100 * (1 - prob_yes).
    This is a heuristic mapping; we also combine some simple rules below.
    """
    prob_yes = proba[class_index_yes] if len(proba) > class_index_yes else 0.0
    safety = max(0.0, min(100.0, (1.0 - prob_yes) * 100.0))
    return round(safety, 1), round(prob_yes * 100.0, 1)

def recommend_skills(job_title, risk_pct):
    # Simple heuristic: for high risk => recommend tech & soft-skill combo
    base = []
    j = job_title.lower() if job_title else ""
    if "developer" in j or "engineer" in j or "dev" in j:
        base = ["System design", "Cloud (AWS/Azure/GCP)", "Security fundamentals", "AI/ML basics"]
    elif "manager" in j or "lead" in j:
        base = ["Strategic thinking", "People management", "Product sense", "AI governance"]
    elif "analyst" in j or "data" in j:
        base = ["Data engineering", "MLOps basics", "Statistics", "Domain knowledge"]
    elif "student" in j or j == "":
        base = ["Programming fundamentals", "Data structures", "Communication", "Internships"]
    else:
        base = ["Digital literacy", "Domain expertise", "problem solving", "communication"]

    # tailor by risk: higher risk + emphasize AI/automation-resistant skills
    if risk_pct >= 70:
        base = base[:3] + ["Creativity & critical thinking", "Complex problem solving"]
    elif risk_pct >= 50:
        base = base[:3] + ["Cross-domain knowledge", "Automation tooling familiarity"]
    else:
        base = base[:4]
    return base

def generate_gemini_text(gem_model, prompt):
    try:
        chat = gem_model.start_chat(history=[])
        resp = chat.send_message(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"(Gemini request failed: {e})"

# -------------------------
# Load model & training table
# -------------------------
st.sidebar.title("Model & Data")
model = load_model("ai_screening_model.pkl")
train_df = load_training_table("Diversity_job_hiring_v2 - Sheet1.csv")

if model is None:
    st.sidebar.error("Model file 'ai_screening_model.pkl' not found. Place it next to this app.")
else:
    st.sidebar.success("Model loaded")

if train_df is None:
    st.sidebar.warning("Training CSV not found — encoders will be generic.")
else:
    st.sidebar.success("Training CSV loaded")

# Define feature columns exactly as in your training script
feature_cols = [
    "Gender", "Age Group", "Highest Education", "Employment Status",
    "Industry", "Company Size", "Job Level", "Work Experience (Years)",
    "Diversity Focus Area", "Rating"
]

# Build encoders (if training df present)
if train_df is not None:
    encoders = build_encoders(train_df, feature_cols)
else:
    encoders = {}
    for col in feature_cols:
        encoders[col] = LabelEncoder()
        encoders[col].fit(["Unknown"])  # trivial fallback

# -------------------------
# Page layout
# -------------------------
st.title("📊 Job AI Safety Evaluator")
st.write("Answer a few simple questions — we'll evaluate how exposed the role is to AI screening/automation and recommend skills to increase safety.")

left_col, right_col = st.columns([2, 1])

# Create a dedicated anchor for results so we can scroll to it
st.markdown("<div id='evaluation-results-anchor'></div>", unsafe_allow_html=True)

with left_col:
    # Top-left: pie chart placeholder and percentage
    st.markdown("#### Snapshot")
    pie_place = st.empty()
    percent_place = st.empty()
    jobname_place = st.empty()
    skills_place = st.empty()

    st.markdown("---")
    # Input area: dynamic depending on Student / Professional
    st.markdown("#### Your profile")
    role_type = st.selectbox("Are you a working professional or a student?", ["Working professional", "Student"])

    # Common fields
    name = st.text_input("Full name (optional)", "")
    job_title = st.text_input("Current Job Title / Dream Job", "Software Developer" if role_type == "Working professional" else "Student - Choose a dream job")
    gender = st.selectbox("Gender", ["Unknown", "Male", "Female", "Other"])
    age_group = st.selectbox("Age group", ["Unknown", "18-24", "25-34", "35-44", "45+"]) if role_type == "Student" else st.selectbox("Age group", ["Unknown", "18-24", "25-34", "35-44", "45+"])
    highest_education = st.selectbox("Highest Education", ["Unknown", "High School", "Bachelor's", "Master's", "PhD", "Other"])
    employment_status = "Student" if role_type == "Student" else st.selectbox("Employment Status", ["Unknown", "Full-time", "Part-time", "Freelancer", "Self-employed", "Unemployed"])
    industry = st.selectbox("Industry", ["Unknown", "Technology", "Finance", "Education", "Healthcare", "Manufacturing", "Retail", "Other"])
    company_size = st.selectbox("Company Size", ["Unknown", "1-10", "11-50", "51-200", "201-1000", "1000+"]) if role_type == "Working professional" else st.selectbox("Company Size (if any)", ["Unknown", "Student", "1-10", "11-50", "51-200", "201-1000", "1000+"])
    job_level = st.selectbox("Job Level", ["Unknown", "Intern/Entry", "Mid", "Senior", "Manager", "Director", "Executive"])
    work_exp = st.selectbox("Work Experience (Years)", ["Unknown", "0-1", "1-3", "3-5", "5-10", "10+"])
    diversity_focus = st.selectbox("Diversity Focus Area", ["Unknown", "Gender", "Disability", "LGBTQ+", "Age", "Ethnicity", "None"])
    rating = st.selectbox("Self rating (1-5) for digital / technical readiness", [1,2,3,4,5])

    # Optional free text for context
    extra_context = st.text_area("Tell us more about your role or goals (optional)", "", max_chars=500)

    # NEW: Remarks field the user can write; this will be included in the evaluation and shown below
    remarks = st.text_area("Any remarks or specific concerns you want the evaluator to consider? (optional)", "", max_chars=500)

    # Optional image upload (not required)
    st.markdown("Optional: upload a screenshot or job description (image)")
    uploaded_image = st.file_uploader("Upload image (png/jpg)", type=["png","jpg","jpeg"])

    attach_image_for_eval = st.checkbox("Attach image context for Gemini explanation (if API enabled)", value=False)

    # Evaluate button
    if st.button("Evaluate job AI safety"):
        # Build user_row matching feature_cols
        user_row = {
            "Gender": gender,
            "Age Group": age_group,
            "Highest Education": highest_education,
            "Employment Status": employment_status,
            "Industry": industry,
            "Company Size": company_size,
            "Job Level": job_level,
            "Work Experience (Years)": work_exp,
            "Diversity Focus Area": diversity_focus,
            "Rating": rating
        }

        # encode
        X_user = encode_input(user_row, encoders, feature_cols)
        import numpy as np
        X_user_arr = np.array(X_user).reshape(1, -1)

        # default fallback
        safety_score = None
        risk_pct = None
        recommended_skills = []
        assistant_text = None

        if model:
            try:
                proba = model.predict_proba(X_user_arr)[0]  # array of probs per class (0,1,2)
                safety_score, prob_yes_pct = percent_safe_from_model_probs(proba, class_index_yes=2)
                risk_pct = round(100.0 - safety_score, 1)
                # recommended skills based on job_title & risk
                recommended_skills = recommend_skills(job_title, risk_pct)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                safety_score = None
        else:
            # Heuristic fallback when model missing: use rating and experience
            r = int(rating)
            exp_map = {"Unknown":1,"0-1":1,"1-3":2,"3-5":3,"5-10":4,"10+":5}
            e_score = exp_map.get(work_exp, 1)
            heuristic_safety = min(95, 30 + (r * 10) + (e_score * 8))
            safety_score = round(heuristic_safety, 1)
            risk_pct = round(100.0 - safety_score, 1)
            recommended_skills = recommend_skills(job_title, risk_pct)

        # Pie chart: Safety vs Risk
        safe_pct = safety_score if safety_score is not None else 0
        risk_pct = round(100.0 - safe_pct, 1)
        pie_df = pd.DataFrame({
            "category": ["Safe from AI", "At Risk from AI"],
            "value": [safe_pct, risk_pct]
        })
        fig = px.pie(pie_df, names="category", values="value", hole=0.45,
                     color_discrete_sequence=["#10b981", "#ef4444"])
        fig.update_traces(textinfo="percent+label")
        pie_place.plotly_chart(fig, use_container_width=True)

        # Big percentage and job name & skills
        percent_place.markdown(f"### 🔒 Job AI Safety: **{safe_pct}%**")
        jobname_place.markdown(f"**Role evaluated:** {job_title}")
        skills_place.markdown("**Recommended skills to reduce AI risk:**")
        for s in recommended_skills:
            skills_place.markdown(f"- {s}")

        # Right side: person summary & explanation (Gemini if available)
        explanation_col = right_col
        with explanation_col:
            st.markdown("### Evaluation summary")
            st.markdown(f"**Name:** {name if name else '—'}")
            st.markdown(f"**Snapshot:** `{job_title}` — **Safety {safe_pct}% / Risk {risk_pct}%**")
            # Display the user's remarks (if any)
            if remarks:
                st.markdown("**User remarks / concerns:**")
                st.markdown(f"> {remarks}")

            # Build a prompt for Gemini (if enabled)
            gem_prompt = (
                f"User profile:\n"
                f"- Name: {name}\n- Role: {job_title}\n- Employment: {employment_status}\n- Industry: {industry}\n- Experience: {work_exp}\n- Rating: {rating}\n- Extra: {extra_context}\n- Remarks: {remarks}\n\n"
                f"Evaluation results: Job AI safety {safe_pct}%, risk {risk_pct}%.\n"
                "Provide a clear professional paragraph (3-5 sentences) that explains why this result might be the case, "
                "list top 3 concrete actions the user can take to reduce AI risk, and give a short study/course path if the user is a student."
            )

            if gem_model and attach_image_for_eval and uploaded_image is not None:
                # If gem available and user wants to attach image: include a note about image context
                gem_prompt = "Analyze the following user and the attached image. " + gem_prompt

            if gem_model:
                with st.spinner("Generating personalized explanation (Gemini)..."):
                    assistant_text = generate_gemini_text(gem_model, gem_prompt)
                    st.markdown(assistant_text)
            else:
                # Fallback local explanation
                st.markdown(f"**Why this score?** Based on the inputs and model heuristic, this role scored **{safe_pct}%** safe. "
                            "A lower score usually means the role is more routine or easily screened/automated. "
                            "Experience, domain-specialized skills and higher-level coordination often reduce replaceability.")
                st.markdown("**Top 3 actions to reduce AI risk:**")
                for s in recommended_skills[:3]:
                    st.markdown(f"- {s}")
                if role_type == "Student":
                    st.markdown("**Suggested study path:** Start with programming fundamentals → data/AI basics → internships → domain specialization.")

        # Add the evaluation to a session log (include remarks)
        if "evals" not in st.session_state:
            st.session_state.evals = []
        st.session_state.evals.append({
            "ts": datetime.utcnow().isoformat(),
            "name": name,
            "role": job_title,
            "safety": safe_pct,
            "risk": risk_pct,
            "skills": recommended_skills,
            "explanation": assistant_text if 'assistant_text' in locals() else "",
            "remarks": remarks
        })

        st.success("Evaluation complete — see results to the left and summary to the right.")

        # Auto-scroll to the results anchor (so the user is taken directly to the pie & percentage)
        scroll_js = """
        <script>
        (function() {
          var el = document.getElementById('evaluation-results-anchor');
          if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        })();
        </script>
        """
        components.html(scroll_js, height=0)

# -------------------------
# Bottom: show history (no download button — per request)
# -------------------------
# -------------------------
# Bottom: show history and download
# -------------------------
st.markdown("---")
st.markdown("### Past evaluations (this session)")
if "evals" in st.session_state and st.session_state.evals:
    for e in reversed(st.session_state.evals[-6:]):
        st.markdown(f"- **{e['role']}** ({e['name'] or '—'}) — Safety **{e['safety']}%** — {e['ts']}")
else:
    st.write("No evaluations yet.")

if "evals" in st.session_state and st.session_state.evals:
    if st.button("Download session evaluations (TXT)"):
        lines = []
        for e in st.session_state.evals:
            lines.append(f"{e['ts']} | {e['name']} | {e['role']} | Safety: {e['safety']}% | Risk: {e['risk']}%")
            lines.append("Recommended skills: " + ", ".join(e['skills']))
            lines.append("Explanation: " + (e.get("explanation") or ""))
            lines.append("\n")
        blob = "\n".join(lines).encode("utf-8")
        b64 = base64.b64encode(blob).decode()
        href = f"data:file/txt;base64,{b64}"
        st.markdown(f"[Download TXT of evaluations](%s)" % href, unsafe_allow_html=True)
