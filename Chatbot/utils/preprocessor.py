import re
import nltk  # Make sure nltk is imported here too
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# Download NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def create_bag_of_words(self, text, words):
        """Create bag of words representation"""
        sentence_words = nltk.word_tokenize(text)
        bag = [1 if word in sentence_words else 0 for word in words]
        return np.array(bag)
