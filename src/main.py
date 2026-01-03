import pandas as pd
import re

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
    df = load_data()

    # Basic dataset info
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns)

    # Check missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Drop rows with missing values
    df = df.dropna()

    print("\nDataset shape after dropping missing values:", df.shape)
    print("\nSample data:")
    print(df.head())

if __name__ == "__main__":
    main()
      