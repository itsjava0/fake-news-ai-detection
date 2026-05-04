# Fake News Detection Using AI

## Overview
This project aims to detect fake news using machine learning (ML) and natural language processing (NLP) techniques.
The model classifies news as real or fake based on patterns in language with high accuracy.

## Motivation
The spread of disinformation on the internet can cause serious social and political risks in our daily lives. Automated fake news detection systems can help journalists, researchers, and platforms identify potentially false and misleading content.

I chose this project to:
- Apply NLP techniques to a real-world problem
- Understand the full ML lifecycle (data → model → evaluation → deployment)
- Build a project that balances simplicity, explanation, and performance

## Dataset
- **Source:** Public fake news dataset (Kaggle)
- **Size:** ~72k news articles
- **Features:**
  - `title`: The article headline
  - `text`: Full article content
  - `label`: 0 → Real news, 1 → Fake news

## Data Cleaning
- Removed rows with missing titles or text
- Final dataset size after cleaning: ~71,500 samples
- The dataset is focused on political news, which may introduce domain bias

## Tools & Technologies
- Python
- pandas
- scikit-learn
- Natural Language Processing (TF-IDF)

## Machine Learning Pipeline

### 1. Text Preprocessing
- Lowercasing
- Removal of URLs and punctuation
- Removal of stopwords
- TF-IDF vectorization to convert text into numerical features

### 2. Train/Test Split
- 80% training data
- 20% testing data
- Stratified to preserve label balance

### 3. Model
- **Logistic Regression** (primary model)

Chosen for:
- Strong performance on sparse TF-IDF features
- Fast training
- High interpretability

## Results

### Logistic Regression Performance
- **Accuracy: ~95%**

Precision / Recall / F1-score:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Real News (0) | 0.96 | 0.94 | 0.95 |
| Fake News (1) | 0.95 | 0.96 | 0.95 |

### Confusion Matrix
```
[[6599  407]
 [ 282 7020]]
```

The model shows balanced performance across both classes, indicating reliability and low bias.

## Error Analysis
To better understand model behavior, misclassified articles were analyzed. Many errors involved:
- Satirical headlines
- Politically biased language
- Sensational but factual reporting

This demonstrates the difficulty of fake news detection even for humans.

## Sample Prediction
**Input:**
```
"Breaking news: Scientists discover a miracle cure the government doesn't want you to know about."
```
**Output:**
```
Prediction: Fake News
```

## Model Persistence
The trained model and TF-IDF vectorizer are saved using joblib:
```
models/
├── logistic_model.joblib
└── tfidf_vectorizer.joblib
```
This allows:
- Reuse without retraining
- Real-time predictions
- Future deployment (e.g. web app)

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/itsjava0/fake-news-ai-detection.git
cd fake-news-ai-detection
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the program**
```bash
python src/main.py
```

## Limitations
- Dataset is domain-specific (mostly political news)
- TF-IDF ignores word order and deep semantic meaning
- Model may struggle with satire or newly emerging misinformation styles

## Future Improvements
- Use word embeddings (Word2Vec, GloVe)
- Experiment with transformer-based models (BERT)
- Expand dataset to multiple news domains
- Build a web interface (Streamlit)
- Add cross-dataset validation
- Improve explainability with SHAP or feature importance analysis

## Author
**Abdurakhmonov Javokhir** — Aspiring Computer Science student with interests in Machine Learning and AI

## Why This Project Matters
This project demonstrates:
- Practical ML implementation
- Strong understanding of NLP fundamentals
- Ability to evaluate and explain model behaviour
- Clean GitHub workflow and reproducibility
