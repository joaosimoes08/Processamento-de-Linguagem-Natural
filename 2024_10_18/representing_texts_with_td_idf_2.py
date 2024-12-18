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

tfidf_char_vectorizer = TfidfVectorizer(analyzer='char_wb',max_df=0.90,max_features=200000,min_df=0.05,use_idf=True,ngram_range=(1,3))
tfidf_char_vectorizer = tfidf_char_vectorizer.fit(sentences)

tfidf_matrix = tfidf_char_vectorizer.transform(sentences)

dense_matrix = tfidf_matrix.todense()

analyze = tfidf_char_vectorizer.build_analyzer()
print(analyze("To Sherlock Holmes she is always _the_woman."))