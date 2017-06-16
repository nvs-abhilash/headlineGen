import operator
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
    context = width // 4 # approx number of tokenized_words of context

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


def syntactic_filter (tag_word_dict):
    words_set = []
    stopwords = nltk.corpus.stopwords.words('english')
    
    for tag in tag_word_dict.keys():
        if (tag in ['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            for w in tag_word_dict[tag]:
                if w not in stopwords and w.isalnum() and not w.isupper(): 
                    words_set.append(w)
            # print (tag)
            # print (tag_word_dict[tag])
    words_set = set (words_set)
    return words_set


def textRank(tokenized_words, tag_word_dict):
    
    words_set = syntactic_filter(tag_word_dict)

    # Add the vertex to the graph
    graph_dict = {}
    for w in words_set:
        graph_dict[w] = []
    
    graph = Graph(graph_dict)

    # Add the edges
    words = nltk.Text (tokenized_words)
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

    sorted_text_rank = sorted(text_rank_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_text_rank
