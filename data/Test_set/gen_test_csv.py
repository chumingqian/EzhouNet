import json
import os
import pandas as pd

# Folder path containing the .json files
folder_path = './test2024_detection_json'  # <-- Change this to your actual path

data = []
event_count = {}

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            content = json.load(f)
            events = content.get('event_annotation', [])
            wav_filename = filename.replace('.json', '.wav')
            abnormal_found = False

            for event in events:
                event_type = event['type']
                if event_type.lower() != 'normal':
                    abnormal_found = True
                    onset = round(int(event['start']) / 1000, 3)  # convert ms to sec
                    offset = round(int(event['end']) / 1000, 3)

                    # Normalize event_label to 'Crackle' if it's Fine or Coarse Crackle
                    if 'crackle' in event_type.lower():
                        event_label = 'Crackle'
                    else:
                        event_label = event_type

                    # Count event types
                    event_count[event_type] = event_count.get(event_type, 0) + 1

                    data.append([
                        wav_filename, onset, offset, event_label, 'Abnormal', event_type
                    ])

            if not abnormal_found:
                data.append([
                    wav_filename, 'NA', 'NA', 'NA', 'Normal', 'NA'
                ])

# Create DataFrame
columns = [
    'filename', 'onset', 'offset', 'event_label', 'record_bin_label', 'record_abnormal_label'
]
df = pd.DataFrame(data, columns=columns)

# Save to TSV
df.to_csv('test_set_four_cls.tsv', sep='\t', index=False)

# Print event counts
print("Abnormal Event Counts:")
for k, v in event_count.items():
    print(f"{k}: {v}")


"""
Abnormal Event Counts:
Wheeze: 141
Fine Crackle: 169
Stridor: 17
Rhonchi: 62
Wheeze+Crackle: 6
Coarse Crackle: 12


"""