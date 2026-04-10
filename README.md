# AI-Based Real-Time Posture Detection System

## 🚀 Overview
This project detects human posture in real-time using computer vision and machine learning. It also provides analytics through a live dashboard.

## 🧠 Features
- Real-time posture detection using MediaPipe
- Machine Learning model (Random Forest)
- Automatic dataset generation
- Hybrid posture classification (ML + angle)
- Data logging system
- Streamlit dashboard for analytics

## 🏗️ System Architecture
Webcam → MediaPipe → Feature Extraction → ML Model → Prediction → CSV Logging → Dashboard

## 🛠️ Tech Stack
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Streamlit

## 📊 Dashboard
Shows:
- Posture score
- Posture distribution
- Spine angle trends

## ▶️ How to Run

### Step 1
cd posture-ai  
source venv/bin/activate  

### Step 2 (Dashboard)
streamlit run dashboard.py  

### Step 3 (Model)
python posture_detector.py  

## 📈 Output
- Real-time posture classification
- Live analytics dashboard

## 🎯 Future Work
- Deep learning models (LSTM)
- Mobile application
- Multi-user posture tracking
