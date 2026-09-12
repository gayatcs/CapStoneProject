# Data Pipeline — Books Scraper (Module 1)

Scrapes [books.toscrape.com](https://books.toscrape.com), cleans the results, loads them into a normalized SQLite database, and runs a set of SQL queries. All work lives in `Module1.ipynb`.

## Requirements

- Python 3.x
- `requests`, `beautifulsoup4`, `pandas` (installed by the first two cells of the notebook if not already present)

## How to run

1. Open `Module1.ipynb` in Jupyter or VS Code's Jupyter extension.
2. Run all cells top to bottom.
3. This regenerates `books.csv` (raw scraped data) and `books.db` (SQLite database) in this folder — both are build outputs, not required inputs, since the notebook reproduces them from scratch.

The insert cell clears existing rows (`DELETE FROM`) before inserting, so re-running the notebook multiple times is safe and never duplicates data or corrupts the schema.

## Data scraped

- Categories: Travel, Mystery, Historical Fiction
- Pagination across each category is followed automatically until no "next page" link remains
- Result: 69 books across the 3 categories

## Cleaning decisions

- **price → `price_gbp`**: the scraped price string has its currency-symbol prefix stripped and is converted to `float`.
- **star_rating → `rating`**: the text rating (`One`…`Five`) is mapped to an integer 1–5.
- **availability**: rows with missing/unparseable availability text are **dropped** rather than imputed. Availability is a categorical/text field, so median imputation (which only makes sense for numeric fields) doesn't apply here; dropping is the correct fallback. In practice every scraped row parsed cleanly, so no rows were actually dropped by this rule.
- **price_inr**: computed as `price_gbp * 105.50` — a fixed, project-defined baseline rate (**1 GBP = 105.50 INR**), not a live or historical market rate. No date reference or network lookup is used for this conversion.

## Database schema

```
categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL,
      rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))
```

## SQL queries

Five queries are included, collectively covering every required clause:

| Query | Clauses demonstrated |
|---|---|
| Top 5 most expensive in-stock books | `SELECT`/`WHERE`, `ORDER BY`, `LIMIT` |
| Distinct star ratings present | `DISTINCT` |
| Books priced £20–£40 | `BETWEEN` |
| Books rated 4 or 5 stars | `IN` |
| Books with their category name | `JOIN` |

A follow-up cell reads two of these results back with `pd.read_sql`, and separately reproduces the join using `pd.merge` directly on the in-memory `books`/`categories` DataFrames (no SQL). The two outputs are compared and confirmed equivalent.
