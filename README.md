# AMR Prediction & Analysis System

An academic machine learning prototype for predicting Amikacin resistance and analyzing antimicrobial resistance (AMR) patterns.

## Project Features

- Amikacin resistance prediction using Random Forest
- Comparison of Random Forest, XGBoost, SVM, and LightGBM
- Species-wise resistance analysis
- ESKAPE-related species comparison
- Model explainability
- Interactive Streamlit dashboard

## Model Performance

Random Forest evaluation results:
- Accuracy: 95.18%
- ROC-AUC: 0.9634
- F1-score: 0.7463

These results apply to the evaluation performed and do not establish clinical validity.

## Data Analytics Workflow

- Data cleaning and preprocessing
- Feature engineering
- Model comparison and evaluation
- Species-level resistance analysis
- Visualization and interpretation of results

The project involved cleaning and preprocessing a 10,000-sample subset, as recorded in the project description.

## Tech Stack

Python, Pandas, Scikit-learn, Streamlit, Matplotlib, Joblib

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Ensure all required model files and CSV analysis files are available in the project directory.

## Project Status

Work in progress. This is an academic prototype, not a clinical diagnostic tool. Further validation is needed before real-world clinical use.
