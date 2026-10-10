# SMS Spam and Ham Detection Using NLP

A machine-learning project that classifies SMS messages as **spam** (unsolicited or promotional messages) or **ham** (ordinary, legitimate messages). The notebook demonstrates a basic Natural Language Processing (NLP) workflow using text cleaning, stemming, Bag-of-Words feature extraction, and a Multinomial Naive Bayes classifier.

## Project Overview

SMS spam can contain misleading prize claims, paid subscriptions, promotional offers, or suspicious requests. This project uses a labeled collection of SMS messages to train a text classifier that learns patterns associated with spam and non-spam messages.

**Current implementation:** a Jupyter Notebook that preprocesses the dataset, converts messages into numeric features, trains a `MultinomialNB` model, and evaluates it on a held-out test split. The notebook does not currently include a command-line interface, web application, saved model, or deployment workflow.

## Dataset

The project uses `SMSSpamCollection.txt`, a tab-separated dataset with two fields:

| Field | Description |
|---|---|
| `label` | Ground-truth class: `ham` or `spam` |
| `Message` | SMS message text |

The notebook loads **5,572 messages**. Its displayed test split contains **1,115 messages** (20% of the dataset).

Place the dataset file in the project directory so that the notebook can load it using a relative path.

## Workflow

1. **Load data** — read the tab-separated file into a Pandas DataFrame with `label` and `Message` columns.
2. **Clean text** — use a regular expression to replace non-alphabetic characters with spaces and convert text to lowercase.
3. **Remove stop words** — remove English stop words using NLTK.
4. **Stem words** — apply NLTK's `PorterStemmer` to reduce words to their stems.
5. **Create a Bag-of-Words representation** — use `CountVectorizer(max_features=100)` to represent each message with counts from a vocabulary limited to 100 features.
6. **Encode labels** — use `pandas.get_dummies()` to encode the `ham`/`spam` labels. The notebook selects the first dummy column as the target; because Pandas orders these columns alphabetically, this means `ham` is encoded as `1` and `spam` as `0` in the current code.
7. **Split data** — use `train_test_split(test_size=0.20)` to create training and test subsets.
8. **Train the classifier** — fit a `MultinomialNB` model on the training data.
9. **Evaluate** — calculate accuracy and a classification report with precision, recall, and F1-score.

## Model and Results

The notebook's recorded evaluation output is:

| Metric | Recorded result |
|---|---:|
| Accuracy | **95.78%** |
| Test samples | 1,115 |

Per-class metrics from the notebook's classification report (class IDs shown as the model evaluates them):

| Class ID | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `0` (spam) | 0.83 | 0.85 | 0.84 | 143 |
| `1` (ham) | 0.98 | 0.97 | 0.98 | 972 |
| Macro average | 0.90 | 0.91 | 0.91 | 1,115 |
| Weighted average | 0.96 | 0.96 | 0.96 | 1,115 |

These are the outputs saved in the uploaded notebook, not a guarantee of results on a new run. The split has no fixed `random_state`, so results can vary between runs.

## Tech Stack

- Python
- Jupyter Notebook
- Pandas and NumPy
- NLTK (English stop words and Porter stemming)
- scikit-learn (`CountVectorizer`, `train_test_split`, `MultinomialNB`, and evaluation metrics)

## Getting Started

### 1. Clone or download the project

Keep these files in the same directory:

```text
project-folder/
├── README.md
├── Spam and Ham Detection Using NLP(1).ipynb
└── SMSSpamCollection.txt
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy nltk scikit-learn jupyter
```

### 4. Update the dataset path

The original notebook contains a machine-specific Windows file path. Replace that line with a relative path before running it:

```python
import pandas as pd

messages = pd.read_csv(
    "SMSSpamCollection.txt",
    sep="\t",
    names=["label", "Message"]
)
```

### 5. Launch the notebook

```bash
jupyter notebook
```

Open `Spam and Ham Detection Using NLP(1).ipynb` and run the cells from top to bottom. The notebook downloads NLTK's English stop-word corpus when the relevant cell runs; an internet connection may be needed the first time.

## Important Limitations and Recommended Improvements

- **Feature extraction happens before the train/test split.** The notebook calls `fit_transform(corpus)` on the complete dataset before splitting. For a more reliable evaluation, split the raw messages first and fit the vectorizer only on the training messages (ideally through a scikit-learn `Pipeline`).
- **The vocabulary is limited to 100 features.** This is useful for demonstrating Bag-of-Words, but may discard informative terms. Tune `max_features` and compare other representations such as TF-IDF.
- **Label encoding is implicit.** The notebook relies on the first column returned by `get_dummies()`. Explicitly map labels, for example `ham: 0` and `spam: 1`, and report class names consistently to avoid confusion.
- **No fixed random seed is used.** Add `random_state` and consider `stratify=y` so that the split is reproducible and preserves class proportions.
- **Class imbalance matters.** The dataset contains many more ham than spam messages. Review per-class precision, recall, F1-score, and the confusion matrix rather than relying on accuracy alone.
- **Preprocessing can remove useful information.** The regular expression removes digits and punctuation, which can discard clues such as prices, short codes, or phone numbers. Test preprocessing choices rather than assuming they always help.
- **No reusable prediction interface is included.** A useful next step is to save the fitted vectorizer and model, then build a small function or application that accepts a new SMS and returns a human-readable prediction.

## Future Enhancements

- Move preprocessing and vectorization into a scikit-learn `Pipeline` to prevent train/test leakage.
- Compare Multinomial Naive Bayes with Logistic Regression and Linear SVM using cross-validation.
- Tune vectorizer settings and evaluate with a confusion matrix and class-wise metrics.
- Add a reusable prediction function and save the trained pipeline with `joblib`.
- Optionally build a Streamlit or Flask interface for entering a message and viewing the prediction.

## Project Purpose

This project is a practical introduction to text classification and NLP: transforming raw text into numeric features, training a supervised classifier, and evaluating how well it distinguishes spam from legitimate SMS messages.
