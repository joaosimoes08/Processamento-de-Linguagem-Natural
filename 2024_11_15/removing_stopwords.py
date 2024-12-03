import csv
import nltk

filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"
csv_file="../PLN_Ex/Chapter01/stopwords.csv"
with open(csv_file, 'r', encoding='utf-8') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    stopwords = [row[0] for row in reader]

file = open(filename, "r", encoding="utf-8")
text = file.read()

text = text.replace("\n", " ")
words = nltk.tokenize.word_tokenize(text)

words = [word for word in words if word.lower() not in stopwords]

print(words)