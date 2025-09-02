import os
import json

# Define the folder containing the JSON files
folder_path = '/media/respecting_god/S2/05_soundEventDetection/SPRSound_sed/SPRSound-main/Detection/train_detection_json'

# Iterate through all files in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        # Open and read the JSON file
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)

                # Check if 'event_annotation' exists and contains the specified annotation
                for annotation in data.get('event_annotation', []):
                    if annotation.get('type') == 'Wheeze+Crackle':
                        print(f"Found in file: {filename}")
                        break  # Stop checking this file after finding the annotation
                #print(" Done")
            except json.JSONDecodeError:
                print(f"Error reading {filename}. It may not be a valid JSON file.")


print("\n===========================Check ==============")

import os
import json
from collections import defaultdict

# Define the folder containing the JSON files
# folder_path = 'path/to/your/folder'

# Dictionary to count annotation types
type_counts = defaultdict(int)

# Iterate through all files in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        # Open and read the JSON file
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)

                # Check if 'event_annotation' exists
                for annotation in data.get('event_annotation', []):
                    annotation_type = annotation.get('type')
                    type_counts[annotation_type] += 1

                    # Check for the specific annotation type
                    if annotation_type == 'Wheeze+Crackle':
                        print(f"Found in file: {filename}")
                        break  # Stop checking this file after finding the annotation
            except json.JSONDecodeError:
                print(f"Error reading {filename}. It may not be a valid JSON file.")

# Print the counts of each annotation type
print("\nAnnotation Type Counts:")
for annotation_type, count in type_counts.items():
    print(f"{annotation_type}: {count}")