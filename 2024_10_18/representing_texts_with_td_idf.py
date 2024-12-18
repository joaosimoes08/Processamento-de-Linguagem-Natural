import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from removing_stopwords import read_in_csv
from bag_of_words import get_sentences

filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"

stemmer = SnowballStemmer('english')
stopwords_file_path = "../PLN_Ex/Chapter01/stopwords.csv"
sentences = get_sentences(filename)

def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in \
    string.punctuation]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

stopword_list = read_in_csv(stopwords_file_path)
stemmed_stopwords = [tokenize_and_stem(stopword)[0] for stopword in stopword_list]
stopword_list = stopword_list + stemmed_stopwords

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=200000,min_df=0.05, stop_words=stopword_list,use_idf=True,tokenizer=tokenize_and_stem,ngram_range=(1,3))
tfidf_vectorizer = tfidf_vectorizer.fit(sentences)

tfidf_matrix = tfidf_vectorizer.transform(sentences)

dense_matrix = tfidf_matrix.todense()
analyze = tfidf_vectorizer.build_analyzer()
print(analyze("To Sherlock Holmes she is always _the_woman."))