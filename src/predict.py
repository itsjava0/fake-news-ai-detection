def predict_news(model, vectorizer, text):
    text_vector =vectorizer.transform([text])
    prediction=model.predict(text_vector)[0]

    if prediction == 1:
        return "Fake News"
    else:
        return "Real News"