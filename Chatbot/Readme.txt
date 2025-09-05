📋 Project Overview

AI ChatBot is an intelligent conversational assistant built with Python, Natural Language Processing (NLP), and Deep Learning. The chatbot understands user queries, classifies intents, and provides responses either from its trained knowledge base or by searching the internet for real-time information.
Creator: Shirshendu Sekhar Mondal
Email: ronimandal222@gmail.com
GitHub: https://github.com/michael7664
License: MIT License (Copyright © 2024 Shirshendu Sekhar Mondal)




🚀 Features

Intent Recognition: Uses deep learning to classify user queries into predefined categories
Natural Language Processing: Text preprocessing with NLTK for better understanding
Web Search Integration: Automatically searches Google for unknown questions
Real-time Responses: Dynamic answers with context awareness
Web Interface: Beautiful Flask-based web application
Customizable: Easy to add new intents and responses




📦 Installation

Prerequisites:
Python 3.8+
pip package manager


After downloading this project from the git repository, open the folder in Terminal/Command Line
Run these commands:
  1. pip install -r requirements.txt
  2. python train_model.py
  3. python app.py
  4. Open your browser and navigate to: http://localhost:8080 (or the port shown in terminal)




🏗️ Project Structure

chatbot_project/
├── data/
│   └── intents.json          # Training data with patterns and responses
├── models/                   # Trained model files (auto-generated)
├── templates/
│   └── index.html           # Web interface
├── utils/
│   ├── preprocessor.py      # Text cleaning and processing
│   ├── model_trainer.py     # Neural network training
│   ├── response_generator.py # Response handling logic
│   ├── web_search.py        # Internet search functionality
│   └── google_searcher.py   # Google search integration
├── app.py                   # Flask web application
├── main.py                  # Command-line interface
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
└── README.md              # This documentation




⚙️ Customization

Adding New Intents:
Edit data/intents.json to add new capabilities:
{
  "tag": "your_new_intent",
  "patterns": [
    "pattern 1",
    "pattern 2",
    "pattern 3"
  ],
  "responses": [
    "Response 1",
    "Response 2",
    "Response 3"
  ]
}


Retraining the Model:
After modifying intents.json, delete old model files 
rm models/intent_classifier.h5 models/tokenizer.pickle models/label_encoder.pickle

Then retrain again: 
python train_model.py

Modifying Web Interface:
Edit templates/index.html to change the look and feel of the web app.




🔧 Technical Details

Neural Network Architecture:
Input Layer: Bag-of-words representation
Hidden Layers: 3 dense layers with dropout
Output Layer: Softmax classification
Optimizer: Adam with learning rate 0.0005
Regularization: Dropout to prevent overfitting


NLP Processing:
Text cleaning and lowercase conversion
Tokenization with NLTK
Stopword removal
Stemming with Porter Stemmer


Web Search:
Google search integration
Smart snippet extraction
Rate limiting for API respect
Fallback mechanisms




📝 License

MIT License

Copyright © 2024 Shirshendu Sekhar Mondal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


