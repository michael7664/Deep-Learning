import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_trainer import ModelTrainer

def main():
    print("Training chatbot model...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    trainer = ModelTrainer()
    model, history = trainer.train(
        data_path='data/intents.json',
        model_save_path='models/intent_classifier.h5'
    )
    
    print("Model training completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()
