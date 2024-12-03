from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

words = ['leaf', 'leaves', 'booking', 'writing','completed', 'stemming', 'skies']

stemmed_words1 = [stemmer.stem(word) for word in words]

print(stemmed_words1)

stemmer = SnowballStemmer('portuguese')
portuguese_words = ['caminhando', 'bebendo', 'amigo']
stemmed_words2 = [stemmer.stem(word) for word in portuguese_words]

print(stemmed_words2)