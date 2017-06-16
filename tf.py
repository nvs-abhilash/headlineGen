""" Implementation of Tf scores """
import nltk
import sys
from nltk.corpus import reuters
from math import log10
from reuters_tf_idf import tfIdf
import operator

def tf (tokenized_words, tagged_words):
    stopwords = nltk.corpus.stopwords.words('english')

    filtered_text = []
    for w in tokenized_words:
        if w not in stopwords and w.isalnum():
            filtered_text.append(w)
    tfValue = nltk.FreqDist([w for w in filtered_text if not w.isupper()])

    # Calculating tf
    for key, value in tfValue.items():
        value = 1 + log10(value)
        tfValue[key] = value

    sorted_tfValues = sorted(tfValue.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_tfValues