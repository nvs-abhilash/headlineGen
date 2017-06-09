import nltk
import sys
from nltk.corpus import reuters
from math import log10

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

def main(argv):

    fName = './reuters/training/' + str(argv)
    f = open(fName, 'r')
    raw_text = f.read()
    words = nltk.word_tokenize(raw_text)
    text = words

    # words = ['Hello', 'world', 'I', 'am', 'a', 'perfect', 'dog', 'who', 'got', 'killed']
    tagged_words = nltk.pos_tag(words)

    tags = set([tag for (word, tag) in tagged_words])

    tag_dict = {}

    for (word, tag) in tagged_words:
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
        print ('{:>15}\t\t{:>15}'.format(word, str(val)))

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
            print ('{:>15}\t\t{:>15}'.format(word, str(val)))
    set2 = set([w.lower() for (w, val) in sorted_x[:15] if val > 1])


    print ("\n\nIntersection of the top 15 of both: ")
    print (set1.intersection(set2))
    print ("Size of the set: ", end=' ')
    print (len(set1.intersection(set2)))

    print ("\n\nHeading = ", end = ' ')
    print (headings)

    headings = set(headings)
    print ("\n\nIntersection of top 15 TextRank with headings: ")
    print (headings.intersection(set1))
    print ("Size of the set: ", end=' ')
    print ('{0:.2f}'.format(len(headings.intersection(set1)) / len(headings) * 100), end='')
    print ("%")

    print ("\n\nIntersection of top 15 Tf-score with headings: ")
    print (headings.intersection(set2))
    print ("Size of the set: ", end=' ')
    print ('{0:.2f}'.format(len(headings.intersection(set2)) / len(headings) * 100), end='')
    print ("%")

if __name__ == "__main__":
    if (len(sys.argv) == 2):
        main(sys.argv[1])
