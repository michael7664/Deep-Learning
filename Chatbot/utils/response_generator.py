import random
import json
import numpy as np
import nltk
from .preprocessor import TextPreprocessor
from .web_search import web_searcher
import pickle
from datetime import datetime

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class ResponseGenerator:
    def __init__(self, intents_path, model_path):
        self.preprocessor = TextPreprocessor()
        self.intents_path = intents_path
        self.model_path = model_path
        self.context = {}
        self.last_intent = None
        
        # Load intents
        with open(intents_path, 'r') as file:
            self.intents_data = json.load(file)
        
        # Load model artifacts
        try:
            with open('models/tokenizer.pickle', 'rb') as handle:
                self.words = pickle.load(handle)
            with open('models/label_encoder.pickle', 'rb') as handle:
                self.le = pickle.load(handle)
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            self.words = []
            self.le = None
    
    def is_question(self, text):
        """Check if the input is a question"""
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', '?',
                         'explain', 'tell me', 'can you', 'could you', 'would you']
        text_lower = text.lower()
        return any(word in text_lower for word in question_words)
    
    def is_greeting(self, text):
        """Check if the input is a greeting"""
        greetings = ['hello', 'hi', 'hey', 'hola', 'greetings', 'good morning', 
                    'good afternoon', 'good evening', 'howdy', 'sup', 'yo',
                    'what\'s up', 'good day', 'hello there', 'hi there']
        text_lower = text.lower()
        return any(greeting in text_lower for greeting in greetings)
    
    def is_thanks(self, text):
        """Check if the input is a thank you"""
        thanks_words = ['thank', 'thanks', 'appreciate', 'grateful', 'cheers']
        text_lower = text.lower()
        return any(word in text_lower for word in thanks_words)
    
    def is_goodbye(self, text):
        """Check if the input is a goodbye"""
        goodbye_words = ['bye', 'goodbye', 'see you', 'farewell', 'later', 
                        'take care', 'adios', 'ciao', 'so long']
        text_lower = text.lower()
        return any(word in text_lower for word in goodbye_words)
    
    def should_search_web(self, tag, confidence, user_input):
        """Determine if we should search the web"""
        # Always search for general_knowledge questions
        if tag == "general_knowledge":
            return True
        
        # Search for questions that don't match known intents well
        if self.is_question(user_input) and confidence < 0.8:
            return True
        
        # Specific patterns that should trigger web search
        search_patterns = [
            'how to', 'what is', 'who is', 'when did', 'where is',
            'why does', 'explain', 'tell me about', 'can you tell me'
        ]
        
        user_input_lower = user_input.lower()
        if any(pattern in user_input_lower for pattern in search_patterns):
            return True
        
        return False
    
    def predict_intent(self, text):
        """Predict the intent of user input"""
        if not self.words or not self.le:
            print("Model artifacts not loaded properly")
            return None, 0
        
        cleaned_text = self.preprocessor.clean_text(text)
        print(f"Cleaned text: {cleaned_text}")
        
        bag = self.preprocessor.create_bag_of_words(cleaned_text, self.words)
        print(f"Bag of words shape: {bag.shape}")
        
        # Load model and predict
        from tensorflow.keras.models import load_model
        try:
            model = load_model(self.model_path)
        except:
            print("Error loading model")
            return None, 0
        
        prediction = model.predict(np.array([bag]), verbose=0)[0]
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]
        
        tag = self.le.inverse_transform([predicted_index])[0]
        
        print(f"Predicted tag: {tag}, Confidence: {confidence:.4f}")
        
        return tag, confidence
    
    def generate_response(self, tag, user_input=""):
        """Generate response based on predicted intent"""
        # Check if we should use web search
        if self.should_search_web(tag, 0.7, user_input):
            print(f"🔍 Web search triggered for: {user_input}")
            return web_searcher.get_answer(user_input)
        
        # Handle context-based responses
        if self.last_intent == "weather" and any(word in user_input.lower() for word in ['bologna', 'london', 'new york', 'paris', 'tokyo']):
            city = next((word for word in ['bologna', 'london', 'new york', 'paris', 'tokyo'] if word in user_input.lower()), 'there')
            response = f"The weather in {city.title()} is pleasant with mild temperatures. Perfect for outdoor activities!"
            self.last_intent = None
            return response
        
        for intent in self.intents_data['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                
                # Handle dynamic responses
                if '{time}' in response:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    response = response.replace('{time}', current_time)
                
                # Store context for follow-up questions
                if tag == "weather":
                    self.last_intent = "weather"
                
                return response
        
        # If no intent matched, use web search
        return self.handle_unknown_query(user_input)
    
    def handle_unknown_query(self, query):
        """Handle queries that don't match any known intent"""
        print(f"❓ Unknown query, searching web: {query}")
        return web_searcher.get_answer(query)
    
    def handle_short_message(self, user_input):
        """Handle short messages like greetings, thanks, etc."""
        user_input_lower = user_input.lower()
        
        # Handle greetings
        if self.is_greeting(user_input_lower):
            greetings = [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! Nice to meet you!",
                "Greetings! How may I assist you?",
                "Hello! I'm here to help."
            ]
            return random.choice(greetings)
        
        # Handle thanks
        if self.is_thanks(user_input_lower):
            thanks_responses = [
                "You're welcome!",
                "Happy to help!",
                "Anytime!",
                "My pleasure!",
                "Glad I could assist you!"
            ]
            return random.choice(thanks_responses)
        
        # Handle goodbye
        if self.is_goodbye(user_input_lower):
            goodbye_responses = [
                "Goodbye! Have a great day!",
                "See you later!",
                "Take care!",
                "Farewell! Come back soon!",
                "Bye! It was nice chatting with you!"
            ]
            return random.choice(goodbye_responses)
        
        # Handle very short questions
        if len(user_input.split()) == 1 and self.is_question(user_input):
            return "Could you please provide more details about what you're looking for?"
        
        return None
    
    def process_input(self, user_input):
        """Process user input and generate response"""
        # Check for exit commands first
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            return "Goodbye! Have a great day!"
        
        # Skip empty inputs
        if not user_input.strip():
            return "Please ask me something!"
        
        # Handle short messages (greetings, thanks, etc.)
        short_response = self.handle_short_message(user_input)
        if short_response:
            return short_response
        
        # For very short non-greeting messages, ask for clarification
        if len(user_input.split()) < 2 and not self.is_question(user_input):
            return "I'd love to help! Could you please provide more details or ask a complete question?"
        
        tag, confidence = self.predict_intent(user_input)
        
        # Use web search for questions or low confidence
        if self.is_question(user_input) or confidence < 0.4 or self.should_search_web(tag, confidence, user_input):
            return self.handle_unknown_query(user_input)
        else:
            response = self.generate_response(tag, user_input)
            return response
