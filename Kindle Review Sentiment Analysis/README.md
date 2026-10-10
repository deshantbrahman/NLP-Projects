# Kindle Review Sentiment Analysis

A Natural Language Processing (NLP) and machine learning project that classifies Amazon Kindle product reviews as **positive** or **negative** using review text. The notebook explores text cleaning, Bag of Words (BoW), TF-IDF feature extraction, and Naive Bayes classification.

## Project Name

**Recommended repository name:** `kindle-review-sentiment-analysis`

## Overview

Online reviews provide useful feedback about products, but reading every review manually can be time-consuming. This project demonstrates a basic sentiment-analysis workflow that converts written reviews into numerical features and trains machine-learning models to predict sentiment.

## Dataset

The notebook uses `all_kindle_review.csv`, a Kindle review dataset, and selects these columns:

| Column | Description |
|---|---|
| `reviewText` | Written review text used as the model input |
| `rating` | Star rating used to create the sentiment label |

The working dataset in the notebook contains **12,000 rows** and two selected columns. The notebook reports no missing values in these selected columns.

### Sentiment-label mapping

The original star ratings are converted into a binary target:

- **Negative (`0`)**: ratings below 3 (1- and 2-star reviews)
- **Positive (`1`)**: ratings of 3 or above (3-, 4-, and 5-star reviews)

This means that **3-star reviews are treated as positive** in the current implementation. Change the threshold if neutral reviews should be handled separately or classified as negative.

## Workflow

1. **Load and inspect data** — read the CSV and select `reviewText` and `rating`.
2. **Create sentiment labels** — map star ratings to binary labels (`0` and `1`).
3. **Preprocess text** — lowercase the reviews, remove most punctuation/special characters, remove English stop words, remove URLs and HTML tags, normalize whitespace, and apply WordNet lemmatization.
4. **Split the data** — use an 80/20 training/test split.
5. **Extract features** — transform review text into numeric features with CountVectorizer (Bag of Words) and TfidfVectorizer (TF-IDF).
6. **Train models** — fit Gaussian Naive Bayes classifiers on the BoW and TF-IDF representations.
7. **Evaluate** — generate predictions and inspect confusion matrices and accuracy.

## Techniques and Libraries

- **Python** — implementation
- **Pandas** — data loading and manipulation
- **Regular expressions (`re`)** — text cleaning
- **BeautifulSoup (`bs4`)** — removal of HTML markup
- **NLTK** — English stop words and WordNet lemmatization
- **Scikit-learn** — train/test split, vectorizers, Gaussian Naive Bayes, and evaluation metrics
- **Jupyter Notebook** — interactive experimentation

## Results

The notebook's recorded Bag of Words evaluation produced this confusion matrix (rows = actual class, columns = predicted class):

| | Predicted negative (`0`) | Predicted positive (`1`) |
|---|---:|---:|
| Actual negative (`0`) | 511 | 308 |
| Actual positive (`1`) | 692 | 889 |

The corresponding recorded accuracy is **58.33%**.

### Important note about the TF-IDF evaluation

The notebook currently contains a prediction-line bug: the TF-IDF test features are passed to `nb_model_bow` instead of `nb_model_tfidf`. Therefore, the recorded TF-IDF confusion matrix and accuracy (**58.17%**) do **not** represent a valid evaluation of the trained TF-IDF model. Correct the line to:

```python
y_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)
```

Then rerun the prediction and evaluation cells before comparing BoW and TF-IDF results. The reported metrics are notebook outputs, not a guarantee of performance on other data.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/kindle-review-sentiment-analysis.git
cd kindle-review-sentiment-analysis
```

Replace `<your-username>` with your GitHub username after creating the repository.

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn nltk beautifulsoup4 lxml jupyter
```

### 4. Prepare the dataset

Place `all_kindle_review.csv` in the path expected by the notebook:

```text
Kindle Reviews/all_kindle_review.csv
```

If your dataset is stored elsewhere, update the CSV path in the notebook. Ensure you have the right to use and redistribute the dataset before publishing it with the repository.

### 5. Download required NLTK resources

Run this once in Python if the notebook reports missing NLTK resources:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the Kindle review sentiment-analysis notebook and run the cells in order.

## Suggested Repository Structure

```text
kindle-review-sentiment-analysis/
├── Kindle_Review_Sentiment_Analysis.ipynb
├── Kindle Reviews/
│   └── all_kindle_review.csv
├── README.md
└── requirements.txt
```

This is a suggested structure; adjust it to match the files you actually commit. Avoid committing large datasets unless redistribution is permitted and the repository size is appropriate.

## Improvements to Explore

- Fix and rerun the TF-IDF prediction cell before comparing models.
- Compare `MultinomialNB`, which is commonly used as a baseline for count-based text features, with the current Gaussian Naive Bayes implementation.
- Avoid converting sparse vectorizer outputs to dense arrays for larger datasets; use sparse-compatible models where possible.
- Use `stratify=y` and a fixed `random_state` in `train_test_split` for more reproducible, class-balanced experiments.
- Report precision, recall, F1-score, and a normalized confusion matrix in addition to accuracy.
- Consider treating 3-star reviews as neutral or excluding them, depending on the project goal.
- Save the fitted preprocessing/vectorization and model pipeline for reuse on new reviews.
- Add a small prediction script or web interface if deployment is desired.

## Limitations

- The task is binary classification, so the current setup does not distinguish neutral reviews.
- The label definition treats 3-star reviews as positive.
- Results depend on the dataset, preprocessing choices, train/test split, and model settings.
- The notebook's TF-IDF evaluation must be corrected before that model's performance can be assessed reliably.
- The notebook demonstrates experimentation; it does not currently include a deployed application or prediction API.

## Author

**Deshant Brahman**

## License and Dataset Attribution

Add a license that matches your intended use, and cite the original dataset source and its terms before publishing the dataset or redistributing it. No dataset source URL or license is specified in the notebook itself.
