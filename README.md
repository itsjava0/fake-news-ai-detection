# Fake News Detection Using AI

## Overview
This project aims to detect fake news using machine learning(ML) and natural language processing(NLP) techniques. 
The model classifies news as real or fake based on patterns in language with high accuracy.

## Motivation
THe spead of disinformation on the internet might cause serious social and political risks in our daily lives. Automated fake news detecton systems can help journalists, researchers and platforms find out potentially false and misleading content

I chose this project to:
-Apply NLP techniques to a real-world problem
-Understand the full ML lifecycle (data->model->evaluation->deployment)
-Build a project that balances simplcity, explanation and performance

## Dataset
Source: Public fake news dataset (Kaggle)
Size: ~72k news articles
Features:
    -title: The article headline
    -text: full article content
    -label: 0->Real news; 1->Fake news
# Data Cleaning
-Removed rows with missing titles or text
-Final dataset size after cleaning: ~71,5k samples
(The dataset is focused on political news, which may introduce domain bias)

## Tools & Technologies
- Python
- pandas
- scikit-learn
- Natural Language Processing (TF-IDF)

## Machine learning pipeline
1) Text processing
-Lowercasing
-Removal of stopwords
-TF-IDF vectorization to convert text into numerical features 
2) Train-test split
-80% training data
-20% testing data
-Startified to preserve label balance
3) Model used 
-Baseline model: Traditional classifier(for comparisson)
-Logistic regression (primary model)
Chosen for:
    -Strong performance on sparse TF-IDF(Term frequency-Inverse document frequency, basically converting text into numbers) features
    -Fast training
    -High interpretability

## Results
Logistic regression performance
-Accuracy: ~95-96%
    How it looks:

        Precision / Recall / F1-score:

            Real news (0):
            -Precision:    0.96
            -Recall:       0.95

            Fake News (1):
            -Precision:    0.96
            -Recall:       0.95
Confusion matrix
        [[6189 892]
        [ 806 6421]]

The model shows balanced performance across both classes, indicating reliabality and low bias

## Error analysis
To better understand model behavior, misclassified articles were analyzed. Many errors involved, including:
    -Satirical headlines
    -Politically biased language
    -Sensational but factual reporting
This demonstrates the difficulty of fake news detection even for humans

## Sample Prediction
Input:
"Breaking news: Scientists dicover a miracle cure the government doesn't want you to know about."
The model output:
"Prediction: Fake News"

## Model persistence
The trained model and TF-IDF vectorizer are saved using joblib:
    models/
    --->logistic_model.joblib
    --->tfidf_vectorizer.joblib

This allows:
    -Reuse witout retraining
    -Read-time predictions
    -Future deployment(for example: web app)

## HOW TO RUN THE PROJECT
1-Clone the repository
    bash:
        git clone https://github.com/itsjava0/fake-news-ai-detection.git
        cd fake-news-ai-detextion
2-Create and activate virtual environment
    bash:
        python -m venv venv
        venv\Scripts\activate # Windows
3-Install dependencies
    bash:
        pip install -r requirements.txt
4-Run the programm
    bash:
        python src/main.py

## Limitations
-Dataset is domain-specific (mostly political news)
-TF-IDF ignores word order and deep semantic meaning
-Model may struggle with satire or newly emerging misinformation styles

## Future improvements
-Use word embeddings (Word2Vec, GloVe)
-Experiment with transformer-based models(BERT)
-Expand dataset to multiple news domains
-Build a web interface (Streamlit)
-Add cross-dataset validation
-Improve explainability with SHAP or feature importance analysis

## AUTHOR: 
    Abdurakhmonov Javokhir
Aspiring Computer Science student with interests in Machine Learning and AI

# Why this project matters
This project demonstrates:
-Practical ML implementatiom=n
-Strong understanding of NLP fundamentals
-Ability to evaluate and explain model behaviour
-Clean GitHub workflow and reproducibility

