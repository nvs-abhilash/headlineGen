import nltk
import sys
from nltk.corpus import reuters
from math import log10
from reuters_tf_idf import tfIdf

from graphs import Graph
from textRank import textRank
from tf import tf


def content_fraction (text):
    stopwords = nltk.corpus.stopwords.tokenized_words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)



# fract = content_fraction (nltk.corpus.reuters.tokenized_words())

def printTraversal (sorted_positions, offset_dict, text_rank_dict, tid_word_dict):
    
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
        while toFind <= 15:
            list2 = offset_dict[tid_word_dict[toFind]]
            
            minDis = maxVal
            for num in list2:
                if abs(num - prev) < abs(minDis):
                    minDis = num - prev
                    minPos = num

            out_list[i].append(minDis)

            prev = minPos
            toFind += 1
            # print ("To find: " + str(toFind) + ", Current pos: " + str(minPos))
        i += 1

    return out_list

def minDist (list1, list2):
    
    if len (list1) > 0 and len (list2) > 0:
        minVal = abs(list1[0] - list2[0])
    else:
        minVal = 100000
    
    for x in list1:
        for y in list2:
            if abs(x - y) < minVal:
                minVal = abs(x - y)
    return minVal

def printMatrix(offset_dict):
    wordsList = {}
    for w in offset_dict.keys():
        for q in offset_dict.keys():
            if w not in wordsList.keys():
                wordsList[w] = {}
                wordsList[w][q] = minDist (offset_dict[w], offset_dict[q])
            else:
                wordsList[w][q] = minDist (offset_dict[w], offset_dict[q])
    
    print ('\n\n', end='\t')
    
    for w in wordsList.keys():
        print ('{:>10}'.format(w), end=' ')
    print ('\n')

    for w in wordsList.keys():
        print ('{:>10}'.format(w), end = ' ')
        for q in wordsList[w].keys():
            print ('{:>10}'.format(wordsList[w][q]), end = ' ')
        print()
        
def printResult (sorted_table, word_tag_dict, offset_dict):
    for (word, val) in sorted_table[:15]:
        print ('{:>15}\t\t{:>15.2f}'.format(word, val), end='\t\t')
        print (word_tag_dict[word.lower()])
    print ('\n\n')
    for (word, val) in sorted_table[:15]:
        print ('{:>15}\t\t{:>15.2f}'.format(word, val), end='\t\t')
        print (offset_dict[word.lower()])

def printTable (sorted_table, offset_dict):
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

    print ("\n\nSorted positions: TF")
    for (pos, tid) in sorted_positions:
        print ("{:>15}\t\t{:>15}\t\t{:>15}".format(pos, tid, tid_word_dict[tid]))
    
    
    out_list = printTraversal(sorted_positions, offset_dict, score_dict, tid_word_dict)
    
    intersectVal = set(out_list[0])
    for lists in out_list:
        print (lists)
        intersectVal = intersectVal.intersection(set(lists))
        
    print ('\n\n')
    print ("Intersection of all the values: ")
    print (intersectVal)

    print ('\n')

def main(argv):

    fName = './reuters/training/' + str(argv)
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

    words = nltk.Text (tokenized_words)
    doc = nltk.ConcordanceIndex(words)
    
    # Call text Rank
    sorted_text_rank = textRank(tokenized_words, tag_word_dict)
    set1 = set([w.lower() for (w, val) in sorted_text_rank[:15]])  
    offset_dict_text_rank = {}
    for words1 in set1:
        offset_dict_text_rank[words1] = doc.offsets(words1)
    
    # Call tf
    sorted_tfValues = tf(tokenized_words, word_tag_dict)
    set2 = set([w.lower() for (w, val) in sorted_tfValues[:15]])  
    offset_dict_tf = {}
    for words2 in set2:
        offset_dict_tf[words2] = doc.offsets(words2)
    
    # Call tf-idf
    sorted_tf_idf = tfIdf (raw_text, word_tag_dict)
    set3 = set([w for (w, val) in sorted_tf_idf[:15]])  
    offset_dict_tf_idf = {}
    for words3 in set3:
        offset_dict_tf_idf[words3] = doc.offsets(words3)
    
    
    """ Printing the resuts"""
    # print (raw_text)
 
    # print ("\n\nText Rank of the document:")
    # printResult (sorted_text_rank, word_tag_dict, offset_dict_text_rank)
    # printTable (sorted_text_rank, offset_dict_text_rank)
    # printMatrix (offset_dict_text_rank)

    print("\n\nTf Scores of the document:")
    printResult (sorted_tfValues, word_tag_dict, offset_dict_tf)
    printTable (sorted_tfValues, offset_dict_tf)
    printMatrix (offset_dict_tf)

    # print ("\n\nTf-Idf scores of the document: ")
    # printResult (sorted_tf_idf, word_tag_dict, offset_dict_tf_idf)
    # printTable (sorted_tf_idf, offset_dict_tf_idf)
    # printMatrix (offset_dict_tf_idf)


if __name__ == "__main__":
    if (len(sys.argv) == 2):
        main(sys.argv[1])
