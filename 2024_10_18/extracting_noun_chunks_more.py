import spacy
nlp = spacy.load('en_core_web_md')

sentence = "All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind."

doc = nlp(sentence)

for noun_chunk in doc.noun_chunks:
    print(noun_chunk.text)

for noun_chunk in doc.noun_chunks:
    print(noun_chunk.text, "\t", noun_chunk.start, "\t",noun_chunk.end)

for noun_chunk in doc.noun_chunks:
    print(noun_chunk.text, "\t", noun_chunk.sent)

for noun_chunk in doc.noun_chunks:
    print(noun_chunk.text, "\t", noun_chunk.root.text)

other_span = "emotions"
other_doc = nlp(other_span)

for noun_chunk in doc.noun_chunks:
    print(noun_chunk.similarity(other_doc))