import nltk

#filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"
filename = "../PLN_Ex/frase_pt.txt"
file = open(filename, "r", encoding="utf-8")
text = file.read()

text = text.replace("\n", " ")

words = nltk.tokenize.word_tokenize(text)

words_with_pos = nltk.pos_tag(words)

print(words_with_pos)