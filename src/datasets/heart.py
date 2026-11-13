"""Maladies cardiaques : D1 à D4 (responsable : Aziz).

Valeurs "expected" = tableau 2 de l'article : (lignes, positifs, négatifs).
Placer les CSV dans data/raw/heart/ (nom D1.csv ou nom d'origine Kaggle).
"""

DATASETS = {
    "D1": {
        "disease": "heart",
        "name": "Heart Disease Dataset (Comprehensive)",
        "url": "https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final",
        "files": ["D1.csv", "heart_statlog_cleveland_hungary_final.csv"],
        "target": "target",
        "expected": (1190, 629, 561),
        # 11 variables : age, sex, chest pain type, resting bp s, cholesterol,
        # fasting blood sugar, resting ecg, max heart rate, exercise angina, oldpeak, ST slope
    },
    "D2": {
        "disease": "heart",
        "name": "Health care: Heart attack possibility",
        "url": "https://www.kaggle.com/datasets/nareshbhat/health-care-data-set-on-heart-attack-possibility",
        "files": ["D2.csv"],  # renommer : D2 et D3 s'appellent tous deux heart.csv sur Kaggle
        "target": "target",
        "expected": (303, 165, 138),
        # 13 variables (Cleveland UCI) : age, sex, cp, trestbps, chol, fbs, restecg,
        # thalach, exang, oldpeak, slope, ca, thal
    },
    "D3": {
        "disease": "heart",
        "name": "Heart Disease Dataset",
        "url": "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset",
        "files": ["D3.csv"],
        "target": "target",
        "expected": (1025, 526, 499),
        # Mêmes 13 variables que D2. Attention : contient beaucoup de doublons.
    },
    "D4": {
        "disease": "heart",
        "name": "Heart Failure Prediction",
        "url": "https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data",
        "files": ["D4.csv", "heart_failure_clinical_records_dataset.csv"],
        "target": "DEATH_EVENT",
        "expected": (299, 96, 203),
        # 12 variables : age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        # high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time
    },
}
