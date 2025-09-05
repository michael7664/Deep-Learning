📋 Project Overview

AI ChatBot is an intelligent conversational assistant built with Python, Natural Language Processing (NLP), and Deep Learning. The chatbot understands user queries, classifies intents, and provides responses either from its trained knowledge base or by searching the internet for real-time information.
Creator: Shirshendu Sekhar Mondal
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

