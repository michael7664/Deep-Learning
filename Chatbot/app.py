from flask import Flask, render_template, request, jsonify
from utils.response_generator import ResponseGenerator
import os

app = Flask(__name__)

# Initialize the chatbot
try:
    response_generator = ResponseGenerator(
        intents_path='data/intents.json',
        model_path='models/intent_classifier.h5'
    )
    print("Chatbot initialized successfully!")
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    response_generator = None

@app.route('/')
def home():
    """Home page with chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    if response_generator is None:
        return jsonify({'response': 'Chatbot is not initialized properly. Please check the console for errors.'})
    
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({'response': 'Please enter a message.'})
        
        print(f"User: {user_input}")
        response = response_generator.process_input(user_input)
        print(f"Bot: {response}")
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'response': 'Sorry, I encountered an error processing your message.'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'chatbot_initialized': response_generator is not None
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Starting Flask server...")
    print("Open http://localhost:8080 in your browser")  # Changed to 8080
    app.run(debug=True, host='0.0.0.0', port=8080)  # Changed to port 8080
