import re

def clean_text(text):
    """Lowercase, remove URLs, punctuation and extra whitespace from text"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
