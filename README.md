# Logistics Delivery Delay Predictor

This project was developed as part of the OFI Services AI Internship case study.

## Problem
Logistics companies often identify delivery issues only after delays occur.
This project predicts delivery delay risk in advance using historical data,
enabling proactive operational decisions.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit

## Approach
- Merged multiple logistics datasets using Order_ID
- Engineered business-relevant features
- Trained a Logistic Regression model
- Built an interactive Streamlit dashboard

## Features
- Predicts delivery delay probability
- Categorizes risk levels (Low / Medium / High)
- Business-friendly, explainable AI solution

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Train the model:
   python model.py
3. Run the app:
   streamlit run app.py

## Outcome
The solution demonstrates how AI can enable predictive logistics,
improve customer experience, and reduce operational inefficiencies.
