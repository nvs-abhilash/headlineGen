import nltk
import sys
from nltk.corpus import reuters
from math import log10
from reuters_tf_idf import tfIdf

from graphs import Graph

def get_concordance(word, doc, width=75, lines=25):
    """
    Print a concordance for ``word`` with the specified context window.

    :param word: The target word
    :type word: str
    :param width: The width of each line, in characters (default=80)
    :type width: int
    :param lines: The number of lines to display (default=25)
    :type lines: int
    """
    half_width = (width - len(word) - 2) // 2
    context = width // 4 # approx number of words of context

    offsets = doc.offsets(word)
    concordance_list = []

    if offsets:
        lines = min(lines, len(offsets))
        # print("Displaying %s of %s matches:" % (lines, len(offsets)))
        
        token = doc.tokens()
        for i in offsets:
            if lines <= 0:
                break
            left = (' ' * half_width +
                    ' '.join(token[i-context:i]))
            right = ' '.join(token[i+1:i+context])
            left = left[-half_width:]
            right = right[:half_width]
            
            concordance_list.append([left, right])
            
            lines -= 1
    else:
        print("No matches")

    return concordance_list

def content_fraction (text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)


def syntactic_filter (tag_dict):
    words_set = []
    stopwords = nltk.corpus.stopwords.words('english')
    
    for tag in tag_dict.keys():
        if (tag in ['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            for w in tag_dict[tag]:
                if w not in stopwords and w.isalnum() and not w.isupper(): 
                    words_set.append(w)
            # print (tag)
            # print (tag_dict[tag])
    words_set = set (words_set)
    return words_set

# fract = content_fraction (nltk.corpus.reuters.words())

# def printTraversal (sorted_positions, offset_dict, text_rank_dict, tid_word_dict):
    
#     listOfOnes = offset_dict[tid_word_dict[1]]

#     listOfOnes = sorted(listOfOnes)
    
#     for x in listOfOnes:
        
#         prev = x
#         toFind = 2
#         while toFind <= 15:
#             list2 = offset_dict[tid_word_dict[toFind]]
            
#             minDis = 15
#             for num in list2:
#                 if  
#         for 

def main(argv):

    fName = './reuters/training/' + str(argv)
    f = open(fName, 'r')
    raw_text = f.read()

    words = nltk.word_tokenize(raw_text)
    for i in range(len(words)):
        words[i] = words[i].lower()
    text = words

    tagged_words = nltk.pos_tag(words)

    tags = set([tag for (word, tag) in tagged_words])
    words_dict = {}
    tag_dict = {}

    for (word, tag) in tagged_words:
        if word in words_dict.keys():
            words_dict[word.lower()].append(tag)
        else:
            words_dict[word.lower()] = [tag]
        
        if tag in tag_dict.keys():
            tag_dict[tag].append(word)
        else:
            tag_dict[tag] = [word]

    words_set = syntactic_filter(tag_dict)

    # Add the vertex to the graph
    # print (words_set)

    graph_dict = {}
    for w in words_set:
        graph_dict[w] = []
    # print (graph_dict)

    graph = Graph(graph_dict)

    # Add the edges
    words = nltk.Text (words)
    doc = nltk.ConcordanceIndex(words)
    for w in words_set:
        results = get_concordance(w, doc)

        for context in results:
            left = context[0].split()
            right = context[1].split()

            for l in left:
                if l in words_set:
                    graph.add_edge((w, l))
                    graph.add_edge((l, w))

            for l in right:
                if l in words_set:
                    graph.add_edge((w, l))
                    graph.add_edge((l, w))

    # Run the text rank algorithm
    delta = 1
    i = 0
    d = 0.85
    while (delta > 0.0001 and i < 5000):
        
        for v in graph.vertices():
            degree = graph.vertex_degree(v)
            old_rank = graph.text_rank(v)

            sum = 0
            for v2 in graph.adjacency_list(v):
                degree2 = graph.vertex_degree(v2)
                # print ("Degree for " + v2 + " = " + str(degree2))

                tr = graph.text_rank(v2)

                sum += tr / degree2

            value = (1 - d) + d * sum
            graph.set_text_rank(v, value)

            if abs(value - old_rank) < delta:
                delta = abs(value - old_rank)
        i = i + 1

    text_rank_dict = {}
    for v in graph.vertices():
        text_rank_dict[v] = graph.text_rank(v)


    import operator
    sorted_x = sorted(text_rank_dict.items(), key=operator.itemgetter(1), reverse=True)
    # print (sorted_x[:15])

    print (words)

    headings = []
    for w in words:
        if w.isupper():
            headings.append(w.lower())
        
    print ("\n\nText Rank of the document:")
    for (word, val) in sorted_x[:15]:
        print ('{:>15}\t\t{:>15}'.format(word, str(val)), end='\t\t')
        print (words_dict[word.lower()])

    set1 = set([w.lower() for (w, val) in sorted_x[:15]])


    """ Implementation of Tf scores """
    stopwords = nltk.corpus.stopwords.words('english')

    news_text = []
    for w in text:
        if w not in stopwords and w.isalnum():
            news_text.append(w)
    fdist = nltk.FreqDist([w for w in news_text if not w.isupper()])

    print("\n\n\nTf Scores of the document:")
    
    # Calculating tf
    for key, value in fdist.items():
        value = 1 + log10(value)
        fdist[key] = value

    import operator
    sorted_x = sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)

    for (word, val) in sorted_x[:15]:
        if (val > 1):
            print ('{:>15}\t\t{:>15}'.format(word, str(val)), end='\t\t')
            print (words_dict[word.lower()])

    set2 = set([w.lower() for (w, val) in sorted_x[:15] if val > 1])


    print ("\n\nIntersection of the top 15 of both: ")
    print (set1.intersection(set2))
    print ("Size of the set: ", end=' ')
    print (len(set1.intersection(set2)))

    set3 , sorted_tf_idf = tfIdf (raw_text, words_dict)
    print ("\n\nIntersection of the top 15 of both: ")
    print (set1.intersection(set3))
    print ("Size of the set: ", end=' ')
    print (len(set1.intersection(set3)))

    # Finding proximity of the set
    # set1, set2, set3


    """ For TEXT RANK """
    offset_dict = {}
    for words1 in set1:
        offset_dict[words1] = doc.offsets(words1)
    
    sorted_text_rank = sorted(text_rank_dict.items(), key=operator.itemgetter(1), reverse=True)

    i = 1
    text_rank_dict = {}
    for (word, rank) in sorted_text_rank[:15]:
        text_rank_dict[word] = (rank, i)
        i = i + 1

    positions_list = []
    for word in offset_dict.keys():
        offsets = offset_dict[word]
        for ofs in offsets:
            positions_list.append((ofs, text_rank_dict[word][1]))
    
    sorted_positions = sorted(positions_list, key=lambda pos: pos[0])
    # print (text_rank_dict)
    # print (positions_list)

    print ("\n\nSorted positions: Text Rank")
    for (pos, tid) in sorted_positions:
        print ("{:>15}\t\t{:>15}".format(pos, tid))
    
    tid_word_dict = {}
    for word in text_rank_dict.keys():
        tid = text_rank_dict[word][1]
        tid_word_dict[tid] = word

    
    """ For TF """
    offset_dict = {}
    for words2 in set2:
        offset_dict[words2] = doc.offsets(words2)
    
    sorted_tf = sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)

    i = 1
    tf_dict = {}
    for (word, tf) in sorted_tf[:15]:
        tf_dict[word] = (tf, i)
        i = i + 1

    positions_list = []
    for word in offset_dict.keys():
        offsets = offset_dict[word]
        for ofs in offsets:
            positions_list.append((ofs, tf_dict[word][1]))
    
    sorted_positions = sorted(positions_list, key=lambda pos: pos[0])
    # print (text_rank_dict)
    # print (positions_list)

    print ("\n\nSorted positions: TF")
    for (pos, tid) in sorted_positions:
        print ("{:>15}\t\t{:>15}".format(pos, tid))
    
    tid_word_dict = {}
    for word in tf_dict.keys():
        tid = tf_dict[word][1]
        tid_word_dict[tid] = word


    """ For TF-IDF """
    offset_dict = {}
    for words3 in set3:
        offset_dict[words3] = doc.offsets(words3)
    
    
    i = 1
    tf_idf_dict = {}
    for (word, tfidf) in sorted_tf_idf[:15]:
        tf_idf_dict[word] = (tfidf, i)
        i = i + 1

    positions_list = []
    for word in offset_dict.keys():
        offsets = offset_dict[word]
        for ofs in offsets:
            positions_list.append((ofs, tf_idf_dict[word][1]))
    
    sorted_positions = sorted(positions_list, key=lambda pos: pos[0])
    # print (text_rank_dict)
    # print (positions_list)

    print ("\n\nSorted positions: TF-IDF")
    for (pos, tid) in sorted_positions:
        print ("{:>15}\t\t{:>15}".format(pos, tid))
    
    tid_word_dict = {}
    for word in tf_idf_dict.keys():
        tid = tf_idf_dict[word][1]
        tid_word_dict[tid] = word

    # printTraversal(sorted_positions, offset_dict, text_rank_dict, tid_word_dict)



    # for words1 in set1:
    #     for words2 in set2:
    #         if word1 != word2:
    #             doc.off
if __name__ == "__main__":
    if (len(sys.argv) == 2):
        main(sys.argv[1])
