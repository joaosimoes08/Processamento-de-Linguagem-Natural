import spacy
from dividing_into_sentences import read_text_file

filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"

text = read_text_file(filename)

nlp = spacy.load('en_core_web_md')
doc = nlp(text)

for noun_chunk in doc.noun_chunks:
    print(noun_chunk.text)