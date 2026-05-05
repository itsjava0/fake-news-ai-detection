# Model Results

## Overview

So after training the model and running it on the test data, I got pretty good results overall.
The accuracy came out to **95.18%** which I think is really solid for a project like this.
Out of 14,308 test articles, the model got the right answer on almost all of them.

---

## Confusion Matrix

```
[[6599  407]
 [ 282 7020]]
```

Ok so basically what this means is:

- **6599** real news articles were correctly identified as real ✓
- **7020** fake news articles were correctly identified as fake ✓
- **407** real news articles were wrongly called fake (false positives)
- **282** fake news articles slipped through and were called real (false negatives)

The false negatives (282 fake articles that got through) are the more dangerous ones in my opinion because those are the ones that could actually fool someone. But 282 out of 7302 fake articles is still pretty low so I'm happy with that.

---

## Precision and Recall

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Real News | 0.96 | 0.94 | 0.95 |
| Fake News | 0.95 | 0.96 | 0.95 |

**Precision** basically means: out of everything the model called fake, how many were actually fake. I got 0.95 which means 95% of the time when it said "fake" it was right.

**Recall** means: out of all the actual fake articles, how many did the model catch. I got 0.96 which means it caught 96% of all fake news in the test set.

Both numbers are really close to each other which means the model isn't really biased toward one class, it performs pretty evenly on both real and fake articles.

---

## What Went Wrong

### 1. It failed on a brand new real news article
I tested the model on a real article from May 2026 about Maine Governor Janet Mills suspending her Senate campaign. The model said **Fake News** with only 59.66% confidence. It was clearly wrong. I think this happened because the model was trained on older political news (probably from around 2016-2018) so it never saw names like "Graham Platner" or events from 2026. Since TF-IDF just looks at word patterns, if the words are unfamiliar it kind of just guesses.

### 2. Satirical articles were hard
Some articles from satirical news sites like The Onion were getting misclassified as real news. The writing style is actually really clean and professional sounding which I think tricks the model. It learned that fake news uses certain patterns of language and satire doesn't really match those patterns even though it's not real.

### 3. Sensational but real headlines
There were some articles with really dramatic or exaggerated sounding headlines that the model kept flagging as fake. Like a headline saying "Scientists discover massive asteroid heading toward Earth" is real news but sounds exactly like something fake. The model probably learned to associate dramatic language with fake news which makes sense but causes problems here.

### 4. Short articles with barely any text
A few articles in the dataset had almost no text content, just a headline and like one or two sentences. The TF-IDF vectorizer doesn't have much to work with in those cases so the model just kind of guesses. Most of the misclassifications I noticed were on really short articles.

### 5. Political bias in the dataset
The whole dataset is basically political news which means the model probably only really works well for political articles. I tried giving it a sports headline once and the confidence score was really low both ways which shows it just doesn't know what to do with non-political content.

---

## Final Thoughts

Overall I'm really happy with 95% accuracy. The main weakness is that the model is kind of stuck in time — it was trained on a specific set of articles and doesn't handle newer or different types of news very well. If I had more time I would try using a BERT model which actually understands the meaning of words instead of just counting them, and I would also add more diverse training data from different topics and time periods.
