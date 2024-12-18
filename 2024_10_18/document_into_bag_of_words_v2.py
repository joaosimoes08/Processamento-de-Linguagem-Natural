from sklearn.feature_extraction.text import CountVectorizer
from dividing_into_sentences import read_text_file, preprocess_text, divide_into_sentences_nltk

filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"
sherlock_holmes_text = read_text_file(filename)
sherlock_holmes_text = preprocess_text(sherlock_holmes_text)
sentences = divide_into_sentences_nltk(sherlock_holmes_text)

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names())

new_sentence = "And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious andquestionable memory."
new_sentence_vector = vectorizer.transform([new_sentence])
analyze = vectorizer.build_analyzer()
print(analyze(new_sentence))

print(new_sentence_vector)