# 📱 Smartphone Addiction Risk Predictor  (Streamlit)

An end-to-end ML dashboard predicting smartphone addiction risk from daily usage habits.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Cloud
1. Push this folder to GitHub
2. Go to share.streamlit.io → New app
3. Select repo, set main file: app.py
4. Click Deploy — live in ~60 seconds

## Features
- 4-tab layout: Predict · What-if · Model info · Severity levels
- Plotly interactive gauge, SHAP waterfall, comparison chart
- Live what-if simulator with dual gauge
- Personalised recommendations ranked by SHAP impact
- Full feature debug table

## Model
Random Forest (400 trees) · ROC-AUC 0.989 · F1 0.952 · Accuracy 93.2%

*For educational purposes only.*
