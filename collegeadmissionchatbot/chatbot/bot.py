import joblib
import random
import json
import os
from datetime import datetime

class CollegeAdmissionChatbot:
    def __init__(self, model_path, dataset_path, log_path):
        # Load the trained model
        try:
            self.intent_model = joblib.load(model_path)
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
        
        # Load the dataset
        try:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)["dataset"]
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")
        
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Prepare responses based on categories
        self.responses = {item["Category"]: item["Bot Response"] for item in self.dataset}
        
        # Default response when no intent is found
        self.default_response = "I'm sorry, I didn't understand that. Could you please clarify?"

    def get_response(self, user_query):
        try:
            # Predict the intent based on the user query
            intent = self.intent_model.predict([user_query.lower()])[0]
            return self.responses.get(intent, self.default_response)
        except Exception as e:
            print(f"Error predicting intent: {e}")
            return self.default_response
    
    def log_interaction(self, user_query, bot_response):
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": user_query,
            "bot_response": bot_response
        }
        
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error logging interaction: {e}")

    def process_query(self, user_query):
        # Get the bot response
        bot_response = self.get_response(user_query)
        
        # Log the interaction
        self.log_interaction(user_query, bot_response)
        
        return bot_response
