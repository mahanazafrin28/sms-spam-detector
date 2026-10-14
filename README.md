# SMS Spam Detector

An ensemble machine-learning model that classifies SMS messages as **Spam** or **Ham**.

## Dataset
SMS Spam Collection Dataset (5572 messages: 4825 ham, 747 spam)
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: `v1` (label), `v2` (message)

## Model Architecture

| Component | Detail |
|---|---|
| Features | Word TF-IDF (1-2 grams) + Char TF-IDF (3-5 grams) |
| Classifier | Soft-voting Ensemble: SVM + Naive Bayes + Logistic Regression |
| Balancing | SMOTE |
| SVM Calibration | CalibratedClassifierCV (for probability output) |
| Cross-validation | 5-fold Stratified CV |

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.9894 |
| Precision | 0.9545 |
| Recall (spam) | 0.9618 |
| F1 Score | 0.9582 |
| ROC-AUC | 0.9973 |
| 5-Fold CV F1 | 0.9565 ± 0.0147 |

### Confusion Matrix
