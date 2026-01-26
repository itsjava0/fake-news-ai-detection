import pandas as pd
import re
import joblib
import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

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
    df['clean_title']=df['title'].apply(clean_text)
    df['combined_text']=df['clean_title']+" "+df['clean_text']

    #Features and labels 
    X = df['combined_text']
    y= df['label']

    #Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

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

    #Logistic regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)

    y_pred_lr = lr_model.predict(X_test_tfidf)

    #Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy:",round(accuracy * 100,2), "%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nLogistic Regression Performance:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred_lr)*100, 2), "%")
    print(classification_report(y_test, y_pred_lr))

    #Confusin matrix
    cm= confusion_matrix(y_test, y_pred)
    print('Confusion matrix:')
    print(cm)

    #Inspect misclassified samples
    misclassified_idx=y_test[y_test != y_pred].index
    misclassified = pd.DataFrame({
        "text": X_test.loc[misclassified_idx],
        'label': y_test.loc[misclassified_idx],
        'prediction': y_pred[misclassified_idx]
    })

    print('\nSample misclassified articles:')
    print(misclassified.head(3))

    #Save model and vectorizer
    os.makedirs("models", exist_ok=True)

    joblib.dump(lr_model, "models/logistic_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    #Load saved model and vectorizer
    loaded_model=joblib.load('models/logistic_model.joblib')
    loaded_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

    sample_text="Breaking news: Scientists discover a miracle cure the government doesn't want you to know about."

    sample_clean = clean_text(sample_text)
    sample_vec = loaded_vectorizer.transform([sample_clean])

    prediction = loaded_model.predict(sample_vec)[0]

    print("\nSample Prediction:")
    print("Text:", sample_text)
    print('Prediction:', 'fake news' if prediction == 1 else "Real News")



if __name__ == "__main__":
    main()
      