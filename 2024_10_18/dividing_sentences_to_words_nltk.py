import nltk

filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"
file=open(filename, "r", encoding="utf-8")
text=file.read()

text = text.replace("\n", " ")

words = nltk.tokenize.word_tokenize(text)
print(words)
