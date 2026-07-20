# Student Placement Predictor

A machine learning project predicting whether an MBA student will be placed in campus recruitment, based on academic performance, work experience, and specialisation.

**Live demo:** https://student-placement-prediction-pyxwtbdff8xvgftfkrsewd.streamlit.app

---

## Project Overview
Campus placement outcomes depend on a mix of academic performance, work
experience, and specialisation choices. This project builds an end-to-end
machine learning pipeline — from raw data to a deployed, interactive web
app — to predict whether a student is likely to be placed, and to explain
*why* the model made that prediction for a given student.

## Features
- Interactive web app for entering a student's profile and getting an
  instant placement prediction
- Probability score, not just a binary outcome
- Per-prediction explainability: shows the top factors that pushed the
  prediction toward or away from "Placed," based on the model's learned
  coefficients
- Full exploratory data analysis and model comparison notebooks included

## Model Development & Evaluation

Several machine learning models were trained and evaluated to identify the best-performing classifier for student placement prediction.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|---------:|----------:|--------:|---------:|---------:|
| Logistic Regression | *85.45%* | 89.07% | 89.82% | 89.33% | *93.93%* |
| SVM | 83.68% | 85.76% | *91.49%* | 88.42% | 92.15% |
| Random Forest | 82.54% | 85.69% | 89.78% | 87.33% | 92.35% |
| KNN | 81.33% | 83.79% | 90.62% | 86.91% | 88.95% |
| Gradient Boosting | 81.34% | 85.89% | 87.21% | 86.02% | 88.18% |

Based on the evaluation metrics, *Logistic Regression* achieved the highest overall performance and was selected for further optimization through hyperparameter tuning.

## Hyperparameter Tuning

To improve the performance of the selected model, **Logistic Regression** was optimized using **GridSearchCV** with 5-fold cross-validation. The hyperparameters were tuned using the **F1 Score** as the evaluation metric.

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| C | 0.1 |
| Penalty | l2 |
| Solver | liblinear |

### Best Cross-Validation Score

**F1 Score:** **0.8981**

### Best Performing Classifier

Although multiple machine learning models were evaluated and tuned, **Logistic Regression** was selected as the final model because it provided:

- High predictive performance
- Excellent generalization on unseen data
- Fast inference for real-time predictions
- Model interpretability through feature coefficients, allowing the application to explain the factors influencing each prediction

The tuned Logistic Regression model was deployed as an interactive **Streamlit** web application for real-time student placement prediction.

## Application Preview

### Home Page

<img width="901" height="816" alt="image" src="https://github.com/user-attachments/assets/aeb9c7c1-e06e-4931-bffc-93d8e38b27c6" />


### Prediction Result

<img width="858" height="260" alt="image" src="https://github.com/user-attachments/assets/0c140ff7-91e9-469d-8e5a-eb86c4e79c46" />


### Feature Importance

<img width="836" height="462" alt="image" src="https://github.com/user-attachments/assets/86f70919-7fe1-43c3-97ae-1ddff9478776" />


## Tech Stack
- **Language:** Python
- **Data handling:** pandas
- **Modeling:** scikit-learn (Logistic Regression, SVM, Random Forest,
  Gradient Boosting, KNN)
- **Evaluation:** Stratified K-Fold Cross-Validation, GridSearchCV
- **Deployment:** Streamlit, Streamlit Community Cloud
- **Version control:** Git, GitHub

## Dataset
- **Source:** Kaggle — Campus Placement dataset (https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
- **Size:** 215 student records
- **Target variable:** Placement status (Placed / Not Placed)
- **Features:** Gender, 10th/12th/degree percentages, board type, stream,
  degree type, work experience, employability test score, MBA
  specialisation, MBA percentage

## Repository Structure
```
├── data/
│   └── placementdata.csv       # Raw dataset
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb  # EDA, cleaning, encoding, scaling
│   └── 02_model_development.ipynb       # Model comparison, tuning, evaluation
├── models/
│   ├── final_model.pkl             # Trained Logistic Regression model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── label_encoders.pkl          # Fitted LabelEncoders (per binary column)
│   ├── feature_columns.pkl         # Final feature column order
│   └── numerical_cols.pkl          # List of scaled numerical columns
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
└── README.md
```

## Installation
```bash
git clone https://github.com/Prachi883/Student-Placement-Prediction.git
cd Student-Placement-Prediction
pip install -r requirements.txt
```

## Running the App
```bash
streamlit run app.py
```
This opens the app locally at `http://localhost:8501`. Alternatively, use
the live deployed version linked at the top of this README.

## Model Performance
Five models were compared using 5-fold stratified cross-validation, then
tuned via `GridSearchCV`. Logistic Regression was selected as the final
model — it matched the top-performing tuned model (SVM) on F1 score while
remaining fully interpretable.

| Metric      | Test Set Score |
|-------------|-----------------|
| Accuracy    | 0.86            |
| Precision   | ~0.89           |
| Recall      | ~0.90           |
| F1 Score    | ~0.90           |
| ROC-AUC     | ~0.94           |

## Key Findings
- 10th, 12th, and degree academic percentages are the strongest predictors of placement.
- Work experience has a strong positive association with placement.
- MBA percentage showed little association with placement outcome,
  unlike the earlier academic scores — a pattern confirmed by both the
  EDA and the final model's coefficients.
- A linear model (Logistic Regression) performed on par with more complex
  models (Random Forest, Gradient Boosting), and the best-tuned SVM
  independently converged on a linear kernel — suggesting the underlying
  relationship between features and placement is largely linear.

## Limitations
- Small dataset (215 records) — results should be interpreted with
  appropriate caution and are not intended for real admissions/hiring
  decisions.
- A single historical dataset from one context; may not generalize to
  other institutions or regions.
- Associations shown are correlational, not causal.

## Future Improvements
- Collect a larger, more diverse dataset to improve generalizability
- Explore additional features (e.g., extracurriculars, interview scores)
- Add authentication and prediction logging to the deployed app

## Author
**Prachi, Sanskriti Karn**
GitHub: [@Prachi883](https://github.com/Prachi883)
	      [@sanskritikarn(https://github.com/sanskritikarn)]

This project was developed collaboratively by Prachi and Sanskriti Karn.
