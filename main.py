import nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import emoji
import contractions
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Make sure to download the necessary NLTK resources
nltk.download('punkt')         # For tokenization
nltk.download('stopwords')     # For stopwords
nltk.download('wordnet')       # For lemmatization


with open('tokenizer.json') as f:
    tokenizer_json = f.read()
    tokenizer_1 = tokenizer_from_json(tokenizer_json)

loaded_model = load_model('hate_speech_model.h5')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
Regex_tokenizer = RegexpTokenizer(r'\w+')

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text)
    text = contractions.fix(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in Regex_tokenizer.tokenize(text)])
    text = emoji.demojize(text)
    text = str(text.lower())
    return text


def preprocess_raw_text(raw_text, tokenizer, maxlen=33):
    cleaned_text = clean_tweet(raw_text)
    text_sequence = tokenizer.texts_to_sequences([cleaned_text])
    text_padded = pad_sequences(text_sequence, maxlen=maxlen, padding='post')
    return text_padded

def get_result(raw_text):
    preprocessed_text = preprocess_raw_text(raw_text, tokenizer_1)
    prediction = loaded_model.predict(preprocessed_text)
    label_index = np.argmax(prediction[0])
    label = "Hate Speech" if label_index == 1 else "Not Hate Speech"
    return label