# Commands to Run the Project

This document explains how to run and test the **Sonar Rock vs Mine Classification Project** step by step.

---

## 1. Train the Model
Trains the machine learning model on the sonar dataset and saves it.

```terminal
python src/train.py --data data/sonar.csv --output model.pkl

---Predicting
python src/predict.py --file data/sample_input.csv



## 2. **Create a professional README.md**

Your `README.md` is the face of your project. Here’s a **ready-to-use structure** for you:  

```markdown
# Sonar Rock vs Mine Classification 🎯

## 📌 Overview
This is a **Machine Learning project** that predicts whether an object detected by sonar is a **Rock** or a **Mine (explosive)**.  
It uses a dataset of 60 sonar readings per object and trains a classification model.

## ⚙️ Tech Stack
- **Python 3**
- **scikit-learn**
- **pandas**
- **numpy**

## 📂 Project Structure

look for README.md
