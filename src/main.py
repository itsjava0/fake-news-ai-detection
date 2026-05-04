import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import clean_text
from train import train_model
from evaluate import evaluate_model
from predict import predict_news


def load_data():
    """Load and return the fake news dataset."""
    df = pd.read_csv("data/fake_news.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


def main():
    # Load and clean data
    df = load_data()
    df = df.dropna()
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_title"] = df["title"].apply(clean_text)
    df["combined_text"] = df["clean_title"] + " " + df["clean_text"]

    # Features and labels
    X = df["combined_text"]
    y = df["label"]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_df=0.7,
        min_df=5,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train and evaluate model
    lr_model = train_model(X_train_tfidf, y_train)
    evaluate_model(lr_model, X_test_tfidf, y_test)

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_model, "models/logistic_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    # CLI
    print("\n==============================")
    print("   Fake News Detector (CLI)")
    print("==============================")

    while True:
        user_input = input("\nEnter a news article (or type 'exit' to quit):\n> ")

        if user_input.lower() == "exit":
            print("\nExiting... Goodbye!")
            break

        label, confidence = predict_news(lr_model, vectorizer, user_input)

        print("\nPrediction:", label)
        print("Confidence:", confidence, "%")


if __name__ == "__main__":
    main()
