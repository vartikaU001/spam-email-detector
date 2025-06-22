import string
import nltk
import pickle
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
VOCAB_SIZE = 5000

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    stop_words = stopwords.words('english')
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def load_tokenizer_from_file(path='src/tokenizer.pickle'):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
