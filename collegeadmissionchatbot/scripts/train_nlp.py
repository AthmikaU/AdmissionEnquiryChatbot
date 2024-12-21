import json
import os
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fuzzywuzzy import fuzz  # Fuzzy matching for typos and errors
import random

# Load spaCy model for grammar parsing
nlp = spacy.load('en_core_web_sm')

# Markov Chain fallback responses (simplified)
fallback_responses = {
    "general": ["Sorry, I didn't quite get that. Could you please rephrase?"],
    "admissions": ["Can you please clarify your question regarding admissions?"],
    "program": ["Could you please elaborate on the program you're asking about?"],
    "application process": ["Can you specify your question about the application process?"],
    "deadlines": ["Please visit the admissions page for detailed deadlines."],
    "scholarships": ["For scholarships, check the relevant criteria on our website."],
    # Add other categories as needed
}

def preprocess_text(text):
    """
    Preprocesses a single text input using spaCy.
    - Lowercase conversion
    - Lemmatization
    - Stop-word and punctuation removal
    """
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def preprocess_dataset(dataset_path):
    """
    Loads dataset and prepares questions and their associated intents.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)["dataset"]
    questions, intents = [], []
    for item in data:
        category = item["Category"]
        for question in item["User Questions"]:
            clean_question = preprocess_text(question)
            questions.append(clean_question)
            intents.append(category)
    return questions, intents

def fuzzy_match(query, categories, threshold=80):
    """
    Uses fuzzy matching to find the closest matching category to handle typos.
    """
    best_match = None
    highest_score = 0
    for category in categories:
        score = fuzz.ratio(query, category)
        if score > highest_score and score >= threshold:
            highest_score = score
            best_match = category
    return best_match

def train_intent_model(dataset_path, model_output_path):
    """
    Trains the intent model using Support Vector Machine (SVM) with TF-IDF feature extraction.
    """
    questions, intents = preprocess_dataset(dataset_path)
    
    # Ensuring that the test_size is valid and that all classes are represented
    # Increase the test_size to 0.3 (30%) or reduce the number of categories/classes
    X_train, X_val, y_train, y_val = train_test_split(
        questions, intents, test_size=0.3, random_state=42, stratify=intents)
    
    model = Pipeline([ 
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),  # TF-IDF with bigrams
        ('classifier', SVC(kernel='linear'))  # SVM classifier
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Validate the model
    y_pred = model.predict(X_val)
    print("Validation Performance:\n")
    print(classification_report(y_val, y_pred))
    
    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved at {model_output_path}")


def predict_intent(query, model, categories):
    """
    Predict the intent of a user query using the trained model, with fallback logic.
    """
    query_processed = preprocess_text(query)
    intent = model.predict([query_processed])[0]
    
    # Check if model prediction is confident enough or fallback is needed
    if intent not in categories:
        print("Model's intent prediction is not confident. Using fallback response.")
        return random.choice(fallback_responses.get("general", []))
    
    return intent

def get_fallback_response(query, categories):
    """
    If no accurate intent is found, fallback to relevant responses based on fuzzy matching.
    """
    matched_category = fuzzy_match(query, categories)
    if matched_category:
        return random.choice(fallback_responses.get(matched_category, []))
    else:
        return random.choice(fallback_responses.get("general", []))

if __name__ == "__main__":
    dataset_path = 'data/training_data/college_admissions.json'
    model_output_path = 'models/nlp/model_svm_fw.pkl'
    
    # Updated categories from your list
    categories = [
        "Timings", "Safety", "Undergraduate Courses", "Associate Dean of Academics", "Placement Training",
        "Seminar Hall", "Industry Collaborations", "Overview", "Farewell", "Services", "Scholarship Application Process",
        "Greetings", "bye", "Useful Links", "Fee Structure of Hostel", "College Reputation", "Dean of Placements", 
        "Admission Process", "Research Opportunities", "Library Contact", "MBA Admission", "Placement", "Internship", 
        "Internet Details", "Accommodation", "History", "Recommended Resource Links", "Courses", "SOSC Club", 
        "Engineering Fee Structure", "Program Outcomes", "Contact Information", "Bot Information", 
        "Associate Dean of Research and Development", "MBA Entrance Exam", "Student Strength", "College Location", 
        "Director of Research", "Mission", "MBA Application Deadline", "MBA Duration", "Staff Members", "Casual Interaction", 
        "Application Process", "Sahyadri Nihongo Club", "Alumni Association", "Associate Dean of Student Welfare", 
        "Food Court / Canteen", "Principal", "Vision", "Wi-Fi Facilities", "Engineering Programs", 
        "Fees Structure - Management quota", "College Affiliation", "Placement Facilities", "Digital Library", 
        "Dean of Strategic Planning", "Confusion", "Student-Centric Initiatives", "Program Educational Objectives", 
        "Attendance Policy", "Gym", "Engineering Admission Process", "Eligibility Criteria", "Required Documents", 
        "Student Startups", "MBA Part-time", "Director", "Achievements", "Medical Facilities", "Placement Vision and Mission", 
        "Contact Placement Department", "Bot Developer Info", "Campus", "MBA Entrance Syllabus", "MBA Fee Structure", 
        "Bus Service Changes", "VTU Syllabus", "Journals", "Statistics", "VTU Question Paper", "Rules & Regulations", 
        "Vision and Mission", "Sports Facilities", "Student Clubs", "Auditorium", "College Ranking", "Bus Service Refund", 
        "Vice Principal", "NRI Fees Structure", "CEO", "CET Admission", "Mech-Tech Garage", "Infrastructure", 
        "Mentor / Guide / Student Services", "Lateral Entry Admission", "Skill-Focused Learning", "Hostel Timings", 
        "B.E. Syllabus", "Academic Opportunities", "Bus Service Rules", "Bot Capabilities", 
        "HOD of Computer Science and Engineering Department", "Faculty Information", "Comed-K Admission", "Research", 
        "Extracurricular Activities", "Scholarships", "Placements", "MBA Admission Process", "Dean of Career Guidance, Training & Placement", 
        "MBA Placements", "Hostel", "Bus Service", "Innovation and Incubation"
    ]
    
    train_intent_model(dataset_path, model_output_path)
    
    # Load the trained model
    model = joblib.load(model_output_path)
    
    # Example query prediction
    user_query = "Tell me about the admission process."
    intent = predict_intent(user_query, model, categories)
    if intent:
        print(f"Predicted intent: {intent}")
    else:
        fallback_response = get_fallback_response(user_query, categories)
        print(f"Fallback response: {fallback_response}")
