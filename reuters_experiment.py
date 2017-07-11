import nltk
import sys
from nltk.corpus import reuters
from math import log10
from reuters_tf_idf import tfIdf
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize

from graphs import Graph
from textRank import textRank
from tf import tf

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.tokenized_words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)



# fract = content_fraction (nltk.corpus.reuters.tokenized_words())

def print_top_sentence(raw_text, sorted_table, matches, out_list, tid_word_dict, words_list):

    sent_tokenized_list = sent_tokenize(raw_text)

    i = 0
    length = len(sent_tokenized_list)
    while i < length:
        sentence = sent_tokenized_list[i]

        sentences = sentence.split('\n')

        sent_tokenized_list[i] = sentences[0]

        j = 1
        while j < len(sentences):
            sent_tokenized_list.append(sentences[j])
            j += 1
        i += 1

    # Generating the words set
    th = 5
    words_set = set()
    for lists in out_list:
        for i in range(len(lists)):
            if lists[i] <= th and lists[i] >= -th:
                if i > 0:
                    words_set.add((tid_word_dict[i+1], tid_word_dict[i]))

    bag_of_words = set()
    for w in words_list.keys():
        for q in words_list[w].keys():
            if words_list[w][q] <= 5 and words_list[w][q] >= -5:
                bag_of_words.add(w)
                bag_of_words.add(q)

    print ("Bag of words close enough: ")
    for w in bag_of_words:
        print (w)
    print ("\n\n")
    print("Heading:")
    print(sent_tokenized_list[0])

    sent_list = set()
    for sentence in sent_tokenized_list[1:min(5, len(sent_tokenized_list)-1)]:
        count = 0
        for (word1, word2) in words_set:
            if word1 in nltk.word_tokenize(sentence.lower()) and \
                word2 in nltk.word_tokenize(sentence.lower()):
                sent_list.add((count, sentence))
                
    sorted_sentence = sent_list
    print("\n\nTop sentences:")
    for sent in sorted_sentence:
        print(sent[1])
    print('\n\n')

def printTraversal(sorted_positions, offset_dict, text_rank_dict, tid_word_dict):

    listOfOnes = offset_dict[tid_word_dict[1]]

    listOfOnes = sorted(listOfOnes)

    i = 0
    out_list = []

    maxVal = 0

    for values in offset_dict.values():
        for val in values:
            if val > maxVal:
                maxVal = val

    for x in listOfOnes:
        prev = x
        out_list.append([x])

        toFind = 2
        while toFind <= len(text_rank_dict):
            list2 = offset_dict[tid_word_dict[toFind]]

            minDis = maxVal
            for num in list2:
                if abs(num - prev) < abs(minDis):
                    minDis = num - prev
                    minPos = num

            out_list[i].append(minDis)

            prev = minPos
            toFind += 1
        i += 1

    return out_list

def minDist(list1, list2):

    if len(list1) > 0 and len(list2) > 0:
        minVal = list1[0] - list2[0]
    else:
        minVal = 100000

    for x in list1:
        for y in list2:
            if abs(x - y) < abs(minVal):
                minVal = x - y
    return minVal

def printMatrix(offset_dict):
    wordsList = {}

    words = list(offset_dict.keys())

    i = 0
    while i < len(words):
        w = words[i]
        j = i + 1
        while j < len(words):
            q = words[j]
            if w not in wordsList.keys():
                wordsList[w] = {}
            wordsList[w][q] = minDist(offset_dict[w], offset_dict[q])
            j += 1
        i += 1

    print('\n\n', end='\t\t\t\t')

    last = ''
    for w in wordsList.keys():
        print('{:>15}'.format(w), end=' ')
        last = w
    last = list(wordsList[last].keys())
    print ('{:>15}'.format(last[0]))

    arrived = []
    space = 1
    for w in wordsList.keys():
        print('{:>15}'.format(w), end=' ')

        i = 0
        while i < space:
            print('{:>15}'.format(''), end=' ')
            i += 1
        space += 1

        for q in wordsList[w].keys():
            print('{:>15}'.format(wordsList[w][q]), end=' ')
        print()

    return wordsList

def print_sentences(raw_text, sorted_tfValues, tid_word_dict, words_list):
    sent_tokenized_list = sent_tokenize(raw_text)

    i = 0
    length = len(sent_tokenized_list)
    while i < length:
        sentence = sent_tokenized_list[i]

        sentences = sentence.split('\n')

        sent_tokenized_list[i] = sentences[0]

        j = 1
        while j < len(sentences):
            sent_tokenized_list.append(sentences[j])
            j += 1
        i += 1

    stopwords = nltk.corpus.stopwords.words('english')
    i = 0
    for sent in sent_tokenized_list:
        words = sent.split()
        sent_tokenized_list[i] = ''
        for w in words:
            if w not in stopwords:
                sent_tokenized_list[i] += w + ' '
        i += 1

    for (w, val) in sorted_tfValues:
        print (w, end=':\n')
        for sent in sent_tokenized_list:
            words = sent.split()
            if len(words) > 0 and words[0].lower() == w.lower():
                print (sent)
        print()

def printResult(sorted_table, word_tag_dict, offset_dict):

    for (word, val) in sorted_table[:15]:
        print('{:>15}\t\t{:>15.2f}'.format(word, val), end='\t\t')
        print(word_tag_dict[word.lower()])
    print('\n\n')
    for (word, val) in sorted_table[:15]:
        print('{:>15}\t\t{:>15.2f}'.format(word, val), end='\t\t')
        print(offset_dict[word.lower()])

def printTable(sorted_table, offset_dict):
    i = 1
    score_dict = {}
    for (word, value) in sorted_table[:15]:
        score_dict[word] = (value, i)
        i = i + 1

    positions_list = []
    for word in offset_dict.keys():
        offsets = offset_dict[word]
        for ofs in offsets:
            positions_list.append((ofs, score_dict[word][1]))


    sorted_positions = sorted(positions_list, key=lambda pos: pos[0])
    # print (text_rank_dict)
    # print (positions_list)
    tid_word_dict = {}
    for word in score_dict.keys():
        tid = score_dict[word][1]
        tid_word_dict[tid] = word

    print("\n\nSorted positions: TF\n")
    print("{:>15} {:>15} {:>15}".format("Pos Id", "Term Id", "Term"), end='\n\n')

    for (pos, tid) in sorted_positions:
        print("{:>15} {:>15} {:>15}".format(pos, tid, tid_word_dict[tid]))

    print('\n\n')
    out_list = printTraversal(sorted_positions, offset_dict, score_dict, tid_word_dict)

    intersectVal = set(out_list[0])
    for lists in out_list:
        print(lists)
        intersectVal = intersectVal.intersection(set(lists))

    print('\n\n')
    print("Intersection of all the values: ")
    print(intersectVal)

    print('\n')
    return out_list, tid_word_dict

def main(argv, matches = 2):

    fName = 'bbc/politics/' + str(argv)
    f = open(fName, 'r')
    raw_text = f.read()

    # Tokenize the tokenized_words of the text
    tokenized_words = nltk.word_tokenize(raw_text)

    # Making the tokenized_words to lower case
    for i in range(len(tokenized_words)):
        tokenized_words[i] = tokenized_words[i].lower()

    # POS tag the words
    tagged_words = nltk.pos_tag(tokenized_words)

    # Extracting the tags of the text
    tags = set([tag for (word, tag) in tagged_words])
    word_tag_dict = {}
    tag_word_dict = {}

    for (word, tag) in tagged_words:
        if word in word_tag_dict.keys():
            word_tag_dict[word.lower()].append(tag)
        else:
            word_tag_dict[word.lower()] = [tag]

        if tag in tag_word_dict.keys():
            tag_word_dict[tag].append(word)
        else:
            tag_word_dict[tag] = [word]

    words = nltk.Text(tokenized_words)
    doc = nltk.ConcordanceIndex(words)

    stemmer = PorterStemmer()

    # # Call text Rank
    # sorted_text_rank = textRank(tokenized_words, tag_word_dict)
    # set1 = set([w.lower() for (w, val) in sorted_text_rank[:15]])
    # removeList = []
    # for w in set1:
    #     if stemmer.stem(w) != w and stemmer.stem(w) in set1:
    #         removeList.append(w)

    # for w in removeList:
    #     set1.remove(w)

    # sorted_text_rank = [(w, val) for (w, val) in sorted_text_rank[:15] if w not in removeList]

    # offset_dict_text_rank = {}
    # for words1 in set1:
    #     offset_dict_text_rank[words1] = doc.offsets(words1)

    # Call tf
    sorted_tfValues = tf(tokenized_words, word_tag_dict)
    set2 = set([w.lower() for (w, val) in sorted_tfValues[:15]])
    removeList = []
    for w in set2:
        if stemmer.stem(w) != w and stemmer.stem(w) in set2:
            removeList.append(w)

    for w in removeList:
        set2.remove(w)

    sorted_tfValues = [(w, val) for (w, val) in sorted_tfValues[:15] if w not in removeList]

    offset_dict_tf = {}
    for words2 in set2:
        offset_dict_tf[words2] = doc.offsets(words2)

    # # Call tf-idf
    # sorted_tf_idf = tfIdf (raw_text, word_tag_dict)
    # set3 = set([w for (w, val) in sorted_tf_idf[:15]])
    # removeList = []
    # for w in set3:
    #     if stemmer.stem(w) != w and stemmer.stem(w) in set3:
    #         removeList.append(w)

    # for w in removeList:
    #     set3.remove(w)

    # sorted_tf_idf = [(w, val) for (w, val) in sorted_tf_idf[:15] if w not in removeList]

    # offset_dict_tf_idf = {}
    # for words3 in set3:
    #     offset_dict_tf_idf[words3] = doc.offsets(words3)


    """ Printing the resuts"""
    # print (raw_text)

    # print ("\n\nText Rank of the document:")
    # printResult (sorted_text_rank, word_tag_dict, offset_dict_text_rank)
    # printTable (sorted_text_rank, offset_dict_text_rank)
    # printMatrix (offset_dict_text_rank)

    print("\n\nTf Scores of the document:\n")
    printResult(sorted_tfValues, word_tag_dict, offset_dict_tf)
    out_list, tid_word_dict = printTable(sorted_tfValues, offset_dict_tf)
    words_list = printMatrix(offset_dict_tf)
    print_top_sentence(raw_text, sorted_tfValues, matches, out_list, tid_word_dict, words_list)

    print_sentences (raw_text, sorted_tfValues, tid_word_dict, words_list)

    # print ("\n\nTf-Idf scores of the document: ")
    # printResult (sorted_tf_idf, word_tag_dict, offset_dict_tf_idf)
    # printTable (sorted_tf_idf, offset_dict_tf_idf)
    # printMatrix (offset_dict_tf_idf)


if __name__ == "__main__":
    if (len (sys.argv) == 2):
        main(sys.argv[1])
    if (len(sys.argv) == 3):
        main(sys.argv[1], int(sys.argv[2]))
