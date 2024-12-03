import nltk
from nltk.probability import FreqDist

filename = "../PLN_Ex/Chapter01/sherlock_holmes.txt"
file = open(filename, "r", encoding="utf-8")
text = file.read()

text = text.replace("\n", " ")

words = nltk.tokenize.word_tokenize(text)

freq_dist = FreqDist(word.lower() for word in words)
words_with_frequencies = \
[(word, freq_dist[word]) for word in freq_dist.keys()]

sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])

length_cutoff = int(0.02*len(sorted_words))

stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]]

print(stopwords)