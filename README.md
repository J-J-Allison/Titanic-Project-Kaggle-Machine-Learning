# 🚢 Titanic Survival Prediction — Kaggle Competition

> **Final Kaggle Score: 0.77 (77% accuracy)**

A machine learning project tackling the classic [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). This notebook explores 10 classifiers with extensive feature engineering and hyperparameter tuning, achieving a **77% prediction accuracy** on the held-out test set.

---

## 📊 Results

| # | Model | Kaggle Score |
|---|-------|--------------|
| 1 | Random Forest | NaN |
| 2 | Decision Tree | NaN |
| 3 | KNN | NaN |
| 4 | SVC | NaN |
| 5 | Logistic Regression | NaN |
| 6 | Naive Bayes | NaN |
| 7 | XGBoost | NaN |
| 8 | AdaBoost | NaN |
| 9 | Extra Trees | NaN |
| 10 | Gradient Boosting | NaN |
| 11 | Voting Classifier 1 | NaN |
| **12** | **Voting Classifier 2** | **0.77** |

---

## 🔧 Feature Engineering

The bulk of the performance comes from hand-crafted features derived from the raw passenger data:

| Feature | Source | Rationale |
|---------|--------|-----------|
| `Family_Size` | SibSp + Parch + 1 | Combined family metric |
| `Family_Size_Grouped` | Family_Size | Binned into Alone / Small / Medium / Large |
| `Title` | Name field | Extracted Mr, Mrs, Miss, Master, Military, Noble, etc. |
| `Name_Size` | Name length | Longer names correlated with higher social status |
| `Age` (binned) | Age quintiles | Discretized into 5 ordinal categories |
| `Fare` (binned) | Fare quintiles | Discretized into 5 ordinal categories |
| `Cabin_Assigned` | Cabin | Binary flag — having a recorded cabin predicts survival |
| `Cabin` (deck) | Cabin first letter | Deck location on the ship |
| `TicketNumberCounts` | Ticket | Number of passengers sharing the same ticket |
| `TicketLocation` | Ticket prefix | Standardized ticket office / origin codes |

---

## 🏗️ Pipeline Architecture

All preprocessing and modeling is wrapped in scikit-learn `Pipeline` and `ColumnTransformer` objects to ensure consistent transformation across train and test sets.

```
Raw Data
  │
  ├─ Age ──────────────── SimpleImputer (most frequent)
  ├─ Family_Size_Grouped ─ Imputer → OrdinalEncoder
  ├─ Sex, Embarked ─────── Imputer → OneHotEncoder
  └─ Pclass, Cabin_Assigned, Name_Size, Age, Fare, TicketNumberCounts ── Passthrough
  │
  ▼
Model (with GridSearchCV / RandomizedSearchCV)
```

---

## 🤖 Models Trained

All models were tuned via `GridSearchCV` or `RandomizedSearchCV` with stratified cross-validation:

- **Random Forest** — RandomizedSearchCV (n_estimators, max_depth, min_samples_leaf)
- **Decision Tree** — GridSearchCV (criterion, max_depth, min_samples_split/leaf)
- **K-Nearest Neighbors** — GridSearchCV (n_neighbors, weights, algorithm, p)
- **SVC** — RandomizedSearchCV (C, kernel)
- **Logistic Regression** — GridSearchCV (C, solver)
- **Gaussian Naive Bayes** — GridSearchCV (var_smoothing)
- **XGBoost** — GridSearchCV (booster)
- **AdaBoost** — GridSearchCV (n_estimators, learning_rate, base estimator)
- **Extra Trees** — GridSearchCV (n_estimators, max_depth, min_samples_split/leaf)
- **Gradient Boosting** — GridSearchCV (n_estimators, max_depth, learning_rate, min_samples_leaf)
- **Voting Classifiers ×2** — Weighted hard voting combining best estimators

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/titanic-prediction.git
cd titanic-prediction

# Install dependencies
pip install pandas numpy scikit-learn xgboost seaborn matplotlib

# Run the notebook
jupyter notebook titanic.ipynb
```

Place `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/titanic/data) in the project root.

---

## 📁 Project Structure

```
├── titanic.ipynb          # Main notebook
├── train.csv              # Training data (891 passengers)
├── test.csv               # Test data (418 passengers)
└── submissions/           # Generated submission CSVs for each model
    ├── submission_1.csv   # Random Forest
    ├── submission_2.csv   # Decision Tree
    ├── ...
    └── submission_12.csv  # Voting Classifier 2
```

---

## 📚 Tech Stack

- **Python 3**
- **pandas** / **NumPy** — data manipulation
- **scikit-learn** — preprocessing, modeling, tuning
- **XGBoost** — gradient boosting
- **Seaborn** / **Matplotlib** — visualization

---

## 📝 Key Takeaways

1. **Feature engineering drove the score** — extracting titles, binning age/fare, and creating family size groups contributed more than model selection alone.
2. **Ensemble methods outperformed individual models** — the Voting Classifier combining multiple tuned models achieved the best Kaggle score of **0.77**.
3. **Consistency matters** — applying identical transformations to train and test data via pipelines prevented data leakage and ensured reliable predictions.

---

## 📜 License
