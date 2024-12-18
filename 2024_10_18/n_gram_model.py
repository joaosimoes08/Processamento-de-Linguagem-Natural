from sklearn.feature_extraction.text import CountVectorizer
from dividing_into_sentences import read_text_file, preprocess_text, divide_into_sentences_nltk
from bag_of_words import get_sentences, get_new_sentence_vector

filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"

sentences = get_sentences(filename)

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))

X = bigram_vectorizer.fit_transform(sentences)

denseX = X.todense()
print(denseX)

print(bigram_vectorizer.get_feature_names())

new_sentence = "I had seen little of Holmes lately."
new_sentence_vector = \
bigram_vectorizer.transform([new_sentence])

print(new_sentence_vector)
print(new_sentence_vector.todense())

new_sentence1 = " And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory."

new_sentence_vector1 = vectorizer.transform([new_sentence])

print(new_sentence_vector1)
print(new_sentence_vector1.todense())