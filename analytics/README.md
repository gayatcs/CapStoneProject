# Analytics — Titanic EDA & Predictive Modeling

Two notebooks that load the Titanic dataset, explore it, clean it, and build classification and regression models. `01_eda.ipynb` covers profiling, cleaning, and the data story. `02_modeling.ipynb` covers the full predictive pipeline. Run them in order.

## Requirements

- Python 3.x
- `seaborn`, `pandas`, `matplotlib`, `numpy`, `scikit-learn`, `imbalanced-learn`, `joblib`

## How to run

1. Run `01_eda.ipynb` top to bottom — produces `titanic.csv` (raw offline fallback) and `titanic_cleaned.csv` (cleaned data used by the modeling notebook).
2. Run `02_modeling.ipynb` top to bottom — produces `best_titanic_pipeline.pkl`.

If `sns.load_dataset` cannot reach the internet, replace the first load with `pd.read_csv("titanic.csv")` — the raw fallback is committed in this folder.

---

## Part A — EDA (`01_eda.ipynb`)

### Dataset profile

The raw dataset has **891 rows and 15 columns**. Four columns contain missing values:

| Column | Missing count | Missing % |
|--------|--------------|-----------|
| `deck` | 688 | 77.22 % |
| `age` | 177 | 19.87 % |
| `embarked` | 2 | 0.22 % |
| `embark_town` | 2 | 0.22 % |

### Cleaning decisions

| Column | Strategy | Reason |
|--------|----------|--------|
| `age` | Median imputation | 19.87 % missing — within the 5–30 % imputation band; median preferred over mean because the distribution is right-skewed |
| `embarked` / `embark_town` | Drop the 2 affected rows | 0.22 % missing — below the 5 % drop-row threshold; negligible data loss |
| `deck` | Drop the entire column | 77.22 % missing — too sparse to impute reliably |

After cleaning: **889 rows × 14 columns**.

### Univariate findings (age & fare)

- **Age:** IQR = 13.0, bounds = [2.5, 54.5] → **65 outliers**
- **Fare:** IQR = 23.1, bounds = [−26.8, 65.7] → **114 outliers**
- **Fare is right-skewed:** Mean (32.10) > Median (14.45) > Mode (8.05) — the distribution has a long upper tail driven by a small number of very high fares.

### Survival rates by group

All rates computed on the cleaned DataFrame using boolean masking:

| Group | Survival rate |
|-------|--------------|
| Female | 74.04 % |
| Male | 18.89 % |
| 1st class | 62.62 % |
| 2nd class | 47.28 % |
| 3rd class | 24.24 % |
| Female · 1st class | 96.74 % |
| Female · 2nd class | 92.11 % |
| Female · 3rd class | 50.00 % |
| Male · 1st class | 36.89 % |
| Male · 2nd class | 15.74 % |
| Male · 3rd class | 13.54 % |

### Correlation matrix (6 × 6)

Computed on `survived, pclass, age, sibsp, parch, fare` — `adult_male` and `alone` excluded as derived flags. Rendered as a `sns.heatmap`.

The two strongest off-diagonal correlations by absolute value:

1. **pclass ↔ fare: r = −0.55** — higher passenger class (lower pclass number) is associated with higher fares; first-class tickets cost significantly more than third-class.
2. **sibsp ↔ parch: r = 0.41** — passengers travelling with siblings or spouses also tended to travel with parents or children, reflecting family groups booking together.

### Multivariate data story (4 charts)

| Chart | Key finding |
|-------|-------------|
| Bar — survival by sex | Female survival (~74 %) was nearly 4× higher than male (~19 %); consistent with "women and children first" evacuation policy |
| Box — fare by survival | Survivors paid substantially higher median fares, linking wealth and passenger class to survival chance |
| Scatter — age vs fare, coloured by survival | Survivors cluster among higher-fare passengers; age shows little separation between groups, making fare the stronger signal |
| Pairplot — survived / age / fare / pclass | Socioeconomic status (class + fare combination) is the dominant predictor; age distributions of survivors and non-survivors overlap heavily |

### EDA-stage standardization check

`age` and `fare` were z-score standardized on the full cleaned DataFrame using `StandardScaler` as a sanity check only — not fed into the modeling pipeline.

- Before: age mean = 29.3, std = 13.0; fare mean = 32.1, std = 49.7
- After: both columns have mean ≈ 0, std ≈ 1 (confirmed by printed summary and before/after distribution plots)

The standardized columns `age_z` and `fare_z` are saved into `titanic_cleaned.csv` for completeness but are not used as model features.

---

## Part B — Modeling (`02_modeling.ipynb`)

### Train / test split

80/20 stratified split on `survived` → **711 training rows, 178 test rows**. Stratification was used because the target is imbalanced (61.8 % class 0 / 38.2 % class 1); a plain random split risks skewing the class ratio in either fold.

### Preprocessing pipeline

A `ColumnTransformer` wrapped in a `Pipeline` is fit exclusively on `X_train` and applied in transform-only mode to `X_test` — no step ever sees test data during fitting.

| Feature type | Columns | Imputation | Transform |
|---|---|---|---|
| Numeric | `pclass`, `age`, `sibsp`, `parch`, `fare` | Median | `StandardScaler` |
| Categorical | `sex`, `embarked` | Most-frequent | `OneHotEncoder(handle_unknown='ignore')` |

### Classifier results

Three classifiers trained on the identical 711-row split:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | **0.861** |
| Decision Tree | 0.770 | 0.690 | 0.721 | 0.705 | 0.754 |
| **Random Forest** | **0.820** | 0.781 | **0.735** | **0.758** | 0.818 |

ROC curves for all three plotted on a single figure. The Decision Tree was also rendered with `plot_tree` (feature names and class labels shown).

### Imbalance handling comparison

Logistic Regression retrained three ways to compare the effect of imbalance strategies:

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Baseline — no handling | 0.783 | 0.691 | 0.734 |
| `class_weight='balanced'` | 0.718 | 0.750 | 0.734 |
| SMOTE (training fold only) | 0.735 | 0.735 | **0.735** |

SMOTE was applied inside an `imblearn.pipeline.Pipeline` so oversampling is confined to the training fold — no leakage. SMOTE gave the best result: the highest F1 and the most balanced precision–recall trade-off, though the three methods score very similarly on this dataset.

### Hyperparameter tuning (Random Forest)

`GridSearchCV` (5-fold CV, F1 scoring) over `RandomForestClassifier(oob_score=True)`:

| Hyperparameter | Values searched | Best |
|---|---|---|
| `n_estimators` | 100, 200, 300 | **200** |
| `max_depth` | None, 5, 10, 15 | **15** |
| `max_features` | 'sqrt', 'log2' | **'sqrt'** |

Best cross-validated F1: **0.747** — OOB score: **0.807**

### Fare regression

Multivariate `LinearRegression` predicting `fare` from all other available features (excluding `survived`):

| Metric | Value |
|--------|-------|
| MAE | 21.139 |
| RMSE | 41.747 |
| R² | 0.347 |
| Adjusted R² | 0.291 |

The residual plot shows **heteroscedasticity** — residuals fan out at higher predicted fares and several large outliers are present, indicating the model's error variance grows with predicted fare rather than remaining constant.

### Deployment recommendation

**Random Forest** is recommended for deployment. It achieves the highest accuracy (0.820), F1 (0.758), and recall (0.735). Although Logistic Regression has a slightly higher AUC (0.861), Random Forest provides a better overall balance of accuracy, precision, recall, and F1 for practical deployment. The Decision Tree underperforms on all metrics.

### Saved pipeline

The complete fitted pipeline — `ColumnTransformer` preprocessor + `RandomForestClassifier` — is saved to `best_titanic_pipeline.pkl` using `joblib.dump`. It can be loaded and used end-to-end on raw, unpreprocessed data:

```python
import joblib
pipeline = joblib.load("best_titanic_pipeline.pkl")
predictions = pipeline.predict(raw_df)
```

A verification cell in the notebook confirms that reloaded-pipeline predictions exactly match the original.

---

## Files in this folder

| File | Description |
|------|-------------|
| `01_eda.ipynb` | Part A — profiling, cleaning, EDA |
| `02_modeling.ipynb` | Part B — preprocessing, modeling, evaluation |
| `titanic.csv` | Raw offline fallback (891 rows × 15 cols) |
| `titanic_cleaned.csv` | Cleaned dataset from Part A (889 rows × 16 cols, includes `age_z` / `fare_z`) |
| `best_titanic_pipeline.pkl` | Fitted sklearn Pipeline (preprocessing + Random Forest), ready for inference |
