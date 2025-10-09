import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


np.random.seed(42)
n = 2000

age = np.random.randit(21,78,size=n)
income = np.random(np.random.normal(60000,25000,size=np)).clip(8000,300000)
loan