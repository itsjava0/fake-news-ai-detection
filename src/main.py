import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    # Load dataset
    df = pd.read_csv("data/fake_news.csv")
    
    # Drop useless index column
    df = df.drop(columns=["Unnamed: 0"])
    
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)     # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text)         # remove extra spaces
    return text.strip()

def main():
    # Load and clean data
    df = load_data()
    df = df.dropna()
    df["clean_text"]=df["text"].apply(clean_text)

    #Features and labels 
    X = df['clean_text']
    y= df['label']

    #Train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print('Training samples:', X_train.shape[0])
    print('Testing samples:', X_test.shape[0])

    #TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_df=0.7,
        min_df=5,
        stop_words="english"
    )

    X_train_tfidf=vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    #Train model
    model=MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    #Predictions
    y_pred = model.predict(X_test_tfidf)

    #Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy:",round(accuracy * 100,2), "%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
      