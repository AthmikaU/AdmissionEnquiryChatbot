import json

# Load the dataset from the JSON file
dataset_path = 'data/training_data/college_admissions.json'

with open(dataset_path, 'r') as f:
    data = json.load(f)["dataset"]

# Extract unique categories
categories = set()
for item in data:
    categories.add(item["Category"])

# Print the categories
print("Categories in the dataset:")
for category in categories:
    print(category)
