# TitanicProject — Survival Analysis & Modeling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![pandas](https://img.shields.io/badge/Library-pandas-150458)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E)
![seaborn](https://img.shields.io/badge/Viz-seaborn-4c78a8)

A polished, learning-friendly project exploring the classic Titanic dataset. The notebook(s) walk through exploratory data analysis (EDA), thoughtful feature engineering, and baseline-to-better machine learning models to predict passenger survival.

> Goal: Build a clean, reproducible analysis that teaches good habits and produces a well-explained survival prediction model.

---

## Table of Contents
- Overview
- Dataset
- Project Structure
- Quick Start
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Modeling & Evaluation
- Reproducibility
- Results (placeholders)
- Roadmap
- Acknowledgments

---

## Overview
This repository demonstrates a full, end-to-end mini data science workflow:
- Asking clear questions about survival drivers (Who had the best odds? Why?)
- Exploring and visualizing patterns (age, class, fare, family size, embarkation)
- Engineering features that boost predictive signal
- Training and comparing several models with cross-validation
- Communicating findings with concise visuals and metrics

## Dataset
You can use either of the following:
- Kaggle: "Titanic — Machine Learning from Disaster" (requires a Kaggle account)
  https://www.kaggle.com/c/titanic
- Seaborn’s built-in titanic sample (great for quick demos):
  ```python
  import seaborn as sns
  df = sns.load_dataset("titanic")
  ```

Common fields: Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Cabin, Ticket, Name.

## Project Structure
Suggested (adjust to match your repo):
- notebooks/
  - 01_eda.ipynb — Exploratory analysis & visualizations
  - 02_feature_engineering.ipynb — Cleaning, imputations, feature creation
  - 03_modeling.ipynb — Baseline and tuned models, evaluation
- data/
  - raw/ — Original CSVs
  - processed/ — Cleaned features and training splits
- reports/
  - figs/ — Saved plots (PNG/SVG)
- src/ — Optional reusable helpers (preprocessing, metrics)

## Quick Start
Local (recommended):
1) Create environment and install basics:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install pandas numpy seaborn matplotlib scikit-learn jupyter
   ```
2) Launch Jupyter and open a notebook:
   ```bash
   jupyter notebook
   ```
3) If using Kaggle CSVs, place them under data/raw/ and update the paths in the notebook cells.

Tips:
- Use a fixed random_state for reproducible splits.
- Consider creating a requirements.txt once your environment is stable.

## Exploratory Data Analysis (EDA)
Questions to explore:
- Survival by sex, passenger class, and embarkation port
- Does age or fare correlate with survival?
- Family size effects (SibSp + Parch + 1)
- Missingness patterns (Age, Cabin) and how to treat them

Useful plots:
- Countplots for categorical features (sex, pclass, embarked) vs. survival
- KDE/Histogram for age and fare by survival
- Heatmap for correlations
- FacetGrid to combine dimensions

## Feature Engineering
Ideas you can implement:
- Title extraction from Name (Mr, Mrs, Miss, Master → group rare titles)
- FamilySize = SibSp + Parch + 1
- IsAlone = 1 if FamilySize == 1 else 0
- Ticket/Cabin groupings (presence/first char)
- Age and Fare binning (quantiles) for tree-based models
- Imputation strategies: median by Sex x Pclass groups
- One-hot encode categorical variables (Sex, Embarked, Pclass/title buckets)

## Modeling & Evaluation
Start simple, iterate fast:
- Baseline: LogisticRegression
- Tree-based: RandomForest, GradientBoosting, XGBoost/LightGBM (optional)
- Pipelines: StandardScaler (for linear models) + model in a single fit
- Cross-validation: StratifiedKFold (k=5) for stable estimates
- Metrics: Accuracy for comparability, plus F1 and ROC-AUC

Example skeleton:
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler(with_mean=False)),  # if using sparse encodings
    ("clf", LogisticRegression(max_iter=200, n_jobs=None))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(scores.mean(), scores.std())
```

Tips:
- Tune a couple of key hyperparameters; track CV mean ± std.
- Keep a validation set or use nested CV when comparing many models.

## Reproducibility
- Fix random seeds where applicable (numpy, sklearn, model libs)
- Save final features and trained model (joblib) for reuse
- Record library versions:
  ```bash
  pip freeze > requirements.txt
  ```
- Save plots to reports/figs and link them in the notebook or README

## Results (placeholders)
Replace the placeholders after you run the notebooks.

- Best CV Accuracy: … ± …
- Best model: … (e.g., LogisticRegression / RandomForest)
- Top drivers (example): Sex, Pclass, Age, Fare, IsAlone

| Model              | CV Accuracy (mean ± std) | Notes                |
|--------------------|--------------------------|----------------------|
| LogisticRegression | … ± …                    | Baseline             |
| RandomForest       | … ± …                    | Good with FE         |
| GradientBoosting   | … ± …                    | Strong on tabular    |

## Roadmap
- [ ] Add screenshots/plots to reports/figs
- [ ] Publish a concise summary in README with key charts
- [ ] Add a tidy src/ preprocessing pipeline
- [ ] Try gradient boosting and calibrate probabilities
- [ ] Export a predictions.csv for test/holdout

## Acknowledgments
- Kaggle Titanic competition and community kernels for inspiration
- Seaborn/Matplotlib documentation for visualization ideas

> Pro tip: keep the notebooks linear and narrate your thinking. Future you (and reviewers) will thank you.