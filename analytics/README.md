# Analytics Module — Titanic EDA & Predictive Modeling

This module performs exploratory data analysis, data cleaning, and predictive modeling on the Titanic dataset using two sequential Jupyter notebooks.

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Notebooks (in order)**:
   - **Step 1: `01_eda.ipynb`**
     - Loads raw data (`sns.load_dataset('titanic')` or offline fallback `titanic.csv`).
     - Performs data profiling, missing-value treatment, and outlier detection.
     - Generates exploratory visualizations.
     - Saves the cleaned dataset to `titanic_cleaned.csv`.
   - **Step 2: `02_modeling.ipynb`**
     - Loads `titanic_cleaned.csv`.
     - Builds an end-to-end `ColumnTransformer` preprocessing pipeline.
     - Trains and benchmarks classification models (Logistic Regression, Decision Tree, Random Forest).
     - Tests class imbalance techniques (class weighting, SMOTE) and tunes hyperparameters via `GridSearchCV`.
     - Fits a linear regression baseline for fare prediction.
     - Serializes the final trained pipeline to `best_titanic_pipeline.pkl`.

---

## Files Present

| File | Description |
| :--- | :--- |
| [`01_eda.ipynb`](01_eda.ipynb) | Exploratory data analysis, cleaning decisions, and visualizations |
| [`02_modeling.ipynb`](02_modeling.ipynb) | Preprocessing pipelines, model training, evaluation, and pipeline export |
| [`titanic.csv`](titanic.csv) | Raw dataset for offline execution fallback |
| [`titanic_cleaned.csv`](titanic_cleaned.csv) | Cleaned dataset output from Part A |
| [`best_titanic_pipeline.pkl`](best_titanic_pipeline.pkl) | Final serialized scikit-learn pipeline ready for inference |
| [`requirements.txt`](requirements.txt) | Python dependencies for this module |
| [`README.md`](README.md) | Module documentation and execution guide |
