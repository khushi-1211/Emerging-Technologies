from flask import Flask, request, jsonify, render_template
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Import the EnhancedNLPChatBot class
from model import EnhancedNLPChatBot  # Adjust the import path if necessary

# Load the trained chatbot model
chatbot = None
model_path = 'enhanced_chatbot_model.pkl'

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            chatbot = pickle.load(f)
        logging.info("Chatbot model loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading chatbot model: {str(e)}")
else:
    logging.error(f"Model file {model_path} not found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    if chatbot is None:
        return jsonify({'response': 'Chatbot model not loaded. Please check server logs.'}), 500
    
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'response': 'Please provide a message to get a response.'}), 400
    
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
