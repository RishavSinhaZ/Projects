# 📘 Sonar Rock vs Mine Classification Project

This project uses **Machine Learning** to classify whether a sonar signal is reflected from a **Rock** or a **Mine**.

---

## ✅ Project Structure

```plaintext
sonar-rock-vs-mine/
│
├── data/
│   └── sonar.csv         # Dataset (Sonar signals)
│
├── models/
│   ├── sonar_model.pkl   # Trained ML model
│   └── metrics.txt       # Model performance metrics
│
├── src/
│   ├── train.py          # Script to train the model
│   ├── predict.py        # Script to make predictions
│   ├── make_sample.py    # Generates sample input data
│   └── utils.py          # Helper functions (data loading, preprocessing, etc.)
│
├── requirements.txt      # Dependencies
├── commands.md           # Useful commands to run the project
├── .gitignore            # Git ignore file
└── README.md             # Project documentation
```

---

## 🚀 How the Project Works

1. **Dataset** (`sonar.csv`) contains sonar signal readings for rocks and mines.
2. **Training** (`train.py`) loads the dataset, preprocesses it, and trains a classification model.  
   - Output: `sonar_model.pkl` (saved trained model)  
   - Output: `metrics.txt` (performance metrics like accuracy, precision, etc.)
3. **Prediction** (`predict.py`) loads the trained model and predicts if a given input is **Rock** or **Mine**.
4. **Sample Generation** (`make_sample.py`) creates dummy/test input samples for prediction.
5. **Utilities** (`utils.py`) provides helper functions such as loading data, cleaning, and preprocessing.

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-url>
cd sonar-rock-vs-mine
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python src/train.py
```

### 4️⃣ Make Predictions
```bash
python src/predict.py
```

### 5️⃣ Generate Sample Data
```bash
python src/make_sample.py
```

---

## 📊 Outputs

- **`models/sonar_model.pkl`** → Trained model file (used for predictions).  
- **`models/metrics.txt`** → Stores model evaluation results.  
- **Predictions** printed in the terminal when running `predict.py`.  

---

## 📌 Notes
- Modify `train.py` if you want to change algorithms or parameters.  
- `utils.py` keeps the code modular and reusable.  
- Always retrain the model if you modify preprocessing logic.  

---

## ✅ Useful Commands

See `commands.md` for quick references to training, testing, and prediction scripts.
