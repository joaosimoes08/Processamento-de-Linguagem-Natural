import spacy 
import time

start = time.time()

def main():
    #filename = "../PLN_Ex/Chapter01/sherlock_holmes_1.txt"
    filename = "../PLN_Ex/frase_pt.txt"
    file=open(filename, "r", encoding="utf-8")
    text=file.read()

    text = text.replace("\n", " ")

    nlp = spacy.load("pt_core_news_sm")

    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sents]

    print(sentences)


main()
print("%s s" % (time.time() - start))