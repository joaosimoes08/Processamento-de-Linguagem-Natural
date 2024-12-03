import spacy

#filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"
filename = "../PLN_Ex/frase_pt.txt"
file = open(filename, "r", encoding="utf-8")
text = file.read()

text = text.replace("\n", " ")

nlp = spacy.load("pt_core_news_sm")

doc = nlp(text)

words = [token.text for token in doc]
pos = [token.pos_ for token in doc]
word_pos_tuples = list(zip(words, pos))

print(word_pos_tuples)