from preprocess import clean_text

def predict_news(model, vectorizer, text):
    """Return (label, confidence%) for a raw news article string"""
    if not isinstance(text, str) or text.strip() == "":
        return 'Invalid input', 0.0
    
    text=clean_text(text)
    text_vector =vectorizer.transform([text])

    prediction=model.predict(text_vector)[0]
    probabilities=model.predict_proba(text_vector)[0]

    confidence=max(probabilities)

    label='Fake News' if prediction == 1 else "Real News"

    return label, round(confidence*100, 2)