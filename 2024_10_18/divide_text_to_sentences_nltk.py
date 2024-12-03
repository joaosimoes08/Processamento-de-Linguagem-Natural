import nltk

#filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"
filename = "../PLN_Ex/frase_pt.txt"
file=open(filename, "r", encoding="utf-8")
text=file.read()

text = text.replace("\n", " ")

tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
sentences = tokenizer.tokenize(text)
print(sentences)
