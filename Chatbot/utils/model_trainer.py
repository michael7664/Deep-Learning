import json
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from .preprocessor import TextPreprocessor

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ModelTrainer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.le = LabelEncoder()
        
    def load_data(self, file_path):
        """Load and preprocess training data"""
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        patterns = []
        tags = []
        
        for intent in data['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern)
                tags.append(intent['tag'])
        
        return patterns, tags
    
    def prepare_training_data(self, patterns, tags):
        """Prepare data for training"""
        # Preprocess patterns
        cleaned_patterns = [self.preprocessor.clean_text(pattern) for pattern in patterns]
        
        # Create vocabulary
        words = []
        for pattern in cleaned_patterns:
            words.extend(nltk.word_tokenize(pattern))
        
        words = sorted(set(words))
        
        # Prepare training data
        X = []
        y = []
        
        for pattern, tag in zip(cleaned_patterns, tags):
            bag = self.preprocessor.create_bag_of_words(pattern, words)
            X.append(bag)
            y.append(tag)
        
        # Encode labels
        y_encoded = self.le.fit_transform(y)
        
        return np.array(X), np.array(y_encoded), words
    
    def build_model(self, input_dim, output_dim):
        """Build neural network model with better architecture"""
        model = Sequential([
            Dense(64, input_shape=(input_dim,), activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(output_dim, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data_path, model_save_path):
        """Train the model with early stopping"""
        patterns, tags = self.load_data(data_path)
        X, y, words = self.prepare_training_data(patterns, tags)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build and train model
        model = self.build_model(X.shape[1], len(np.unique(y)))
        
        # Add early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=4,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save model and artifacts
        model.save(model_save_path)
        with open('models/tokenizer.pickle', 'wb') as handle:
            pickle.dump(words, handle)
        with open('models/label_encoder.pickle', 'wb') as handle:
            pickle.dump(self.le, handle)
        
        return model, history
