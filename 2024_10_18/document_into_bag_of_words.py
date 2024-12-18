from sklearn.feature_extraction.text import CountVectorizer
from dividing_into_sentences import read_text_file,preprocess_text, divide_into_sentences_nltk

filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"

def get_sentences(filename):
    sherlock_holmes_text = read_text_file(filename)
    sherlock_holmes_text = preprocess_text(sherlock_holmes_text)
    sentences = divide_into_sentences_nltk(sherlock_holmes_text)
    return sentences

def create_vectorizer(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return (vectorizer, X)

sentences = get_sentences(filename)
(vectorizer, X) = create_vectorizer(sentences)

denseX = X.todense()

new_sentence = "I had seen little of Holmes lately."
new_sentence_vector = vectorizer.transform([new_sentence])

print(new_sentence_vector)
print(new_sentence_vector.todense())