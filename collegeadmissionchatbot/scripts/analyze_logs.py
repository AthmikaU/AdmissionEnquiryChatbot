import pandas as pd
import json

def load_logs(log_file):
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f.readlines()]
    return pd.DataFrame(logs)

def analyze_logs(log_file):
    df = load_logs(log_file)
    print("Total Interactions:", len(df))
    
    # Most common queries
    print("\nMost Common Queries:")
    print(df['user_query'].value_counts().head(5))
    
    # Most frequent responses
    print("\nMost Frequent Responses:")
    print(df['bot_response'].value_counts().head(5))

    # Hourly interaction trends
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    print("\nHourly Interaction Trends:")
    print(df['hour'].value_counts().sort_index())

if __name__ == "__main__":
    analyze_logs('logs/chatbot_logs.jsonl')
