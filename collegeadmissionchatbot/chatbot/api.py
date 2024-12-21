from flask import Flask, request, jsonify, render_template
from chatbot.bot import CollegeAdmissionChatbot
import os
from flask_cors import CORS
#pip install flask flask-cors

# Initialize Flask app
app = Flask(__name__, template_folder="../frontend")
CORS(app, resources={r"/api/chat": {"origins": "*"}})

# Load chatbot
chatbot = CollegeAdmissionChatbot(
    model_path='models/nlp/model_svm_fw.pkl',
    dataset_path='data/training_data/college_admissions.json',
    log_path='logs/chatbot_logs.jsonl'
)

# Serve the chatbot UI
@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

# Chatbot API endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query','')
    
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    try:
        bot_response = chatbot.process_query(user_query)
        return jsonify({"query": user_query, "response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
