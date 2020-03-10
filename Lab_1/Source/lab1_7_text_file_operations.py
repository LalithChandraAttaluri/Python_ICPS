# Importing Required Libraries
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import urllib.request
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.stem import LancasterStemmer, SnowballStemmer, WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, ne_chunk, ngrams

# Reading Text from the URL
wikiURL = "https://umkc.app.box.com/s/7by0f4540cdbdp3pm60h5fxxffefsvrw"
openURL = urllib.request.urlopen(wikiURL)

# Assigning Parsed Web Page into a Variable
soup = BeautifulSoup(openURL.read(), "lxml")

# Kill all script and style elements
for script in soup(["script", "style"]):
    # Rip it Off
    script.extract()

# get text
text = soup.body.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = ' '.join(chunk for chunk in chunks if chunk)


# Saving to a Text File
with open('Input.txt', 'w') as text_file:
    text_file.write(str(text.encode("utf-8")))

# Reading from a Text File
with open('Input.txt', 'r') as text_file:
    read_data = text_file.read()

# Tokenization
"""Sentence Tokenization"""
sentence_tokens = sent_tokenize(text)
print("Sentence Tokenization : \n", sentence_tokens)
"""Word Tokenization"""
word_tokens = [word_tokenize(t) for t in sentence_tokens]
print ("Word Tokenization : \n", word_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
print("Lemmatization :\n")
for tok in word_tokens:
    print(lemmatizer.lemmatize(str(tok)))

# Trigram
print("Trigrams :\n")
trigram = []
for x in word_tokens:
    trigram.append(list(ngrams(x, 3)))
print(trigram)

#Calaculating Word Frequency
trigram_final = [item for sublist in trigram for item in sublist]
wordFreq = nltk.FreqDist(trigram_final)
mostCommon = wordFreq.most_common()

#Extract the top 10 of the most repeated trigrams based on their count.
Top_ten_trigrams = wordFreq.most_common(10)
print("Top 10 Trigrams:\n", Top_ten_trigrams, "\n")

#Writiing top 10 trigrams to a text file
with open("top10_trigrams.txt","w") as tri:
    for ti in Top_ten_trigrams:
        tri.write(str("\n"))
        tri.write(str(ti))

#Finding the sentences that contains most repeated trigrams
final_result = []
for st in sentence_tokens:
    for a, b, c in trigram_final:
        for ((d, e, f), length) in wordFreq.most_common(10):
            if(a, b, c == d, e, f):
                final_result.append(st)

print("List of sentences with most repeated trigrams are: \n", final_result)
print("The most repeated Array is: \n ", max(final_result))

