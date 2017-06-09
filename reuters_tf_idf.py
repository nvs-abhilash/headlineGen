import nltk
from nltk.corpus import reuters
from math import log10

text = reuters.words('training/42')

stopwords = nltk.corpus.stopwords.words('english')

news_text = []
for w in text:
    if w not in stopwords and w.isalnum():
        news_text.append(w)
fdist = nltk.FreqDist(w.lower() for w in news_text)

# Calculating tf
for key, value in fdist.items():
    value = 1 + log10(value)
    fdist[key] = value

import operator
sorted_x = sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)

for (word, val) in sorted_x[:15]:
    print ('{:>15}\t\t{:>15}'.format(word, str(val)))
