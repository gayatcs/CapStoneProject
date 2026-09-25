# Capstone Project

A modular end-to-end data engineering, machine learning analytics, and generative AI support assistant project.

---

## 1. Project Structure & Dependency Setup

This project uses a **per-module `requirements.txt`** structure. Each module folder contains its own dependencies to keep environments lightweight, but you can also install everything via the respective directories.

```
CapStoneProject/
├── README.md                          # Root project documentation
├── data_pipeline/                     # Module 1: Web Scraping & SQL Pipeline
│   ├── Module1.ipynb                  # Scraping, cleaning, SQLite insertion, and queries
│   ├── books.db                       # Generated SQLite database
│   ├── README.md                      # Module 1 detailed documentation
│   └── requirements.txt               # Module 1 dependencies
├── analytics/                         # Module 2: EDA & Predictive Modeling
│   ├── 01_eda.ipynb                   # Data profiling, cleaning, outlier detection, visualizations
│   ├── 02_modeling.ipynb              # Preprocessing, classification, regression, pipeline export
│   ├── titanic.csv                    # Raw dataset offline fallback
│   ├── titanic_cleaned.csv            # Cleaned dataset artifact
│   ├── best_titanic_pipeline.pkl      # Saved inference pipeline artifact
│   ├── README.md                      # Module 2 detailed documentation
│   └── requirements.txt               # Module 2 dependencies
└── support_assistant/                 # Module 3: Policy RAG Support Assistant
    ├── main.py                        # FastAPI application with LangGraph RAG pipeline
    ├── llm agent.ipynb                # Development and experimentation notebook
    ├── Dockerfile                     # Container deployment file
    ├── requirements.txt               # Module 3 dependencies
    ├── README.md                      # Module 3 detailed documentation
    └── docs/                          # Policy documents knowledge base (8 txt files)
```

### Dependency Installation

You can install dependencies per module as needed:

```bash
# Module 1: Data Pipeline
pip install -r data_pipeline/requirements.txt

# Module 2: Analytics
pip install -r analytics/requirements.txt

# Module 3: Support Assistant
pip install -r support_assistant/requirements.txt
```

---

## 2. How to Run Each Module End-to-End

### Module 1: Data Pipeline (`/data_pipeline`)
1. Open [`data_pipeline/Module1.ipynb`](data_pipeline/Module1.ipynb) in Jupyter Notebook or VS Code.
2. Execute all cells from top to bottom.
3. **Outputs generated:**
   - `books.csv` (scraped raw data from Books to Scrape).
   - `books.db` (normalized SQLite database containing `categories` and `books` tables with SQL queries executed).

---

### Module 2: Analytics (`/analytics`)
1. **Part A (EDA):** Open and run [`analytics/01_eda.ipynb`](analytics/01_eda.ipynb) top to bottom.
   - Profiles missing values and distributions.
   - Imputes `age`, drops sparse column `deck` and 2 missing rows in `embarked`.
   - Generates `titanic_cleaned.csv` and univariate/multivariate visualizations.
2. **Part B (Modeling):** Open and run [`analytics/02_modeling.ipynb`](analytics/02_modeling.ipynb) top to bottom.
   - Builds preprocessing pipelines with `ColumnTransformer`.
   - Trains and evaluates Logistic Regression, Decision Tree, and Random Forest classifiers.
   - Evaluates class imbalance handling (Baseline, Class Weighting, SMOTE).
   - Tunes Random Forest hyperparameters via `GridSearchCV`.
   - Fits linear regression for fare prediction.
   - Saves final pipeline to `best_titanic_pipeline.pkl`.

---

### Module 3: Support Assistant (`/support_assistant`)

**Option A: Running Locally**
```bash
cd support_assistant
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```
- Interactive API docs (Swagger): `http://localhost:7860/docs`
- Query the endpoint:
```bash
curl -X POST http://localhost:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the refund policy?"}'
```

**Option B: Running with Docker**
```bash
cd support_assistant
docker build -t support-assistant .
docker run -p 7860:7860 support-assistant
```

---

## 3. Design Decisions Summary

### Module 1: Data Pipeline
- **Pagination & Scraping:** Iterates across Travel, Mystery, and Historical Fiction categories on `books.toscrape.com` until no `next` link exists, capturing 69 books.
- **Data Cleaning:** Stripped currency prefixes, mapped text ratings (`One`..`Five`) to integers 1–5, converted GBP to INR using a fixed conversion baseline rate ($1\text{ GBP} = 105.50\text{ INR}$).
- **Database Normalization:** Designed a 2-table schema (`categories` and `books`) linked by `category_id` foreign key. Idempotent table creation and row insertion (`DELETE FROM` prior to insert) prevents duplicates upon re-execution.

### Module 2: Analytics
- **Missing Value Handling:** Imputed `age` using median (skewed numeric distribution, ~19.9% missing); dropped `deck` due to excessive sparsity (~77.2% missing); dropped 2 rows missing `embarked` (<0.3%).
- **Pipeline Architecture:** Built leakage-free `ColumnTransformer` pipelines fitting imputers and scalers exclusively on `X_train`.
- **Model Selection & Imbalance:** Evaluated Logistic Regression, Decision Tree, and Random Forest with SMOTE oversampling. Random Forest was selected as the deployment candidate for superior overall accuracy (0.820) and F1-score (0.758).
- **Serialization:** Full preprocessing and model pipeline exported to `best_titanic_pipeline.pkl` for direct raw-data inference.

### Module 3: Support Assistant
- **RAG Architecture:** 8 domain policy documents chunked paragraph-wise, indexed into an in-memory ChromaDB vector collection using `all-MiniLM-L6-v2` embeddings.
- **LangGraph Routing:** StateGraph classifies queries into `policy_question` (triggers top-3 semantic chunk retrieval) or `general_question` (answers directly or returns out-of-scope notice without retrieval).
- **Reliability & Fallbacks (`MOCK_LLM`):** Supports `MOCK_LLM=1` (default) for deterministic execution and grading without requiring external paid API keys, with simple toggle for live LLM providers.
