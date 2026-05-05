# Model Limitations

This file is just me being honest about what my model can't do or struggles with.
I think it's important to be upfront about this stuff instead of just showing the good results.

---

## 1. It's basically stuck in time

The training data is from a specific period (around 2016-2018 political news).
So if you give it a news article from 2025 or 2026 with new names, new events, or new topics
it has never seen before — it kind of falls apart. I literally tested this and it called a real
2026 news article fake with only 59% confidence. That's basically a coin flip.

---

## 2. Only really works for political news

The entire dataset is political news articles. That's all it knows.
If you give it a sports article, a science article, a celebrity story — it has no idea what to do.
It might still give you an answer but the confidence will be low and it's probably just guessing.
This model is not a general fake news detector, it's more like a political fake news detector.

---

## 3. It doesn't actually understand language

TF-IDF just counts words. It doesn't know what words mean, it doesn't understand sarcasm,
it doesn't understand context, it doesn't know that "The Onion" is a joke website.
Two sentences can mean completely opposite things and TF-IDF would treat them as similar
if they use the same words. A real understanding of language would need something like BERT.

---

## 4. Satire looks like real news to the model

Satirical articles are written in a clean, professional style on purpose — that's what makes
them funny. The model learned that fake news has certain messy or dramatic patterns, so when
it sees well-written satire it often classifies it as real news. There's not really a fix for
this without a completely different approach.

---

## 5. It ignores word order completely

TF-IDF treats every article as just a bag of words — the order doesn't matter at all.
So "the government is not hiding anything" and "the government is hiding everything" would
look pretty similar to this model because they share most of the same words. Obviously those
two sentences mean completely different things. This is a pretty fundamental weakness.

---

## 6. Short articles confuse it

If an article is really short — like just a headline and one sentence — the model doesn't have
enough text to work with. TF-IDF needs a decent amount of words to find patterns. Short articles
tend to get lower confidence scores and are more likely to be misclassified.

---

## 7. It can be fooled pretty easily

Because the model learned specific word patterns from training data, someone who knows how it
works could probably write fake news that avoids those patterns and slips through. Or write
real-sounding language that triggers fake news flags. It's not robust against someone
actively trying to trick it.

---

## 8. No explanation for its decisions

When the model says "Fake News" it can't tell you why. It just gives a label and a confidence
score. There's no way to know which words or phrases triggered the decision. This makes it
hard to trust and hard to debug when it's wrong.

---

## 9. Dataset probably has its own bias

The dataset came from Kaggle and focuses on US political news. This means the model likely
has a political and cultural bias baked into it from the training data. News from other
countries or in different political contexts might get misclassified just because the writing
style or topics are different from what it trained on.

---

## 10. 95% accuracy sounds great but it's not perfect

Out of 14,308 test articles, the model got 689 wrong. At that scale in a real application —
say a platform with millions of articles — that's a huge number of mistakes. A 5% error rate
sounds small but it really isn't when you're talking about labeling news as fake or real.
Getting that wrong has real consequences.

---

## Summary

This model is a solid starting point and works well in a controlled environment with political
news similar to its training data. But it should not be used as a real-world fake news detector
without major improvements — more diverse data, a better model architecture (like BERT),
and some way to explain its decisions.
