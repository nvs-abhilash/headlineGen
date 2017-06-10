import nltk
import os
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

translator = str.maketrans('', '', string.punctuation)

path = './reuters/training/'
token_dict = {}

def tokenize (text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        # stems.append(PorterStemmer().stem(item))
        stems.append (item)
    return stems

def tfIdf (string, words_dict):
    for dirpath, dirs, files in os.walk(path):
        for f in files:
            fName = os.path.join(dirpath, f)
            # print ("fName = ", fName)
            with open(fName) as pearl:
                text = pearl.read()
                token_dict[f] = text.lower().translate(translator)

    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(token_dict.values())
    tfIdf_dict = {}

    response = tfidf.transform([string])
    # print (response)

    feature_names = tfidf.get_feature_names()
    for col in response.nonzero()[1]:
        tfIdf_dict[feature_names[col]] = response[0, col]

    import operator
    sorted_x = sorted(tfIdf_dict.items(), key=operator.itemgetter(1), reverse=True)


    if len(sorted_x) > 15:
        maxVal = 15
    else:
        maxVal = len(sorted_x)
    
    print ("\n\nTf-Idf scores of the document: ")
    for (word, val) in sorted_x[:maxVal]:
        print ('{:>15}\t\t{:>15}'.format(word, str(val)), end='\t\t')
        print (words_dict[word.lower()])

    set3 = set([w for (w, val) in sorted_x[:maxVal]])
    return (set3, sorted_x[:maxVal])

def main():
    tfIdf ('hello world I am NVS Abhilash')

if __name__ == '__main__':
    main()