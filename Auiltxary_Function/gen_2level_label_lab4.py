import os
import json
import pandas as pd

# Define the folder containing the JSON files and the output file
# input_folder = "path/to/your/json/folder"
# output_tsv_file = "output_file.tsv"

# input_folder ="../data/Train_set/train_detection_json"
# output_tsv_file = "train_set.tsv"


# input_folder = "../data/Train_set/valid_detection_json"
# output_tsv_file = "valid_set_2level_lab4.tsv"


input_folder = "../data/Test_set/test_json"
output_tsv_file = "test_set_2level_lab4.tsv"

# # List to store the data extracted from all JSON files
import os
import json
import pandas as pd
import re

# Define the folder containing the JSON files and the output file
# input_folder = "../data/Train_set/train_detection_json"
# output_tsv_file = "train_set_2level_lab4.tsv"

# List to store the data extracted from all JSON files
data = []
record_summary = {}
processed_files = set()

# Iterate through all files in the specified folder
json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
for filename in json_files:
    json_path = os.path.join(input_folder, filename)

    # Extract patient number and recording location from filename using regex
    match = re.match(r"[_\w\d\.]+", filename)
    if match:
        record_name = filename.replace(".json", ".wav")
        processed_files.add(filename)
    else:
        # If filename does not match, log the reason and continue to the next iteration
        print(f"Filename did not match expected pattern: {filename}")
        continue

    # Open and load the JSON file
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
        event_annotations = json_data.get("event_annotation", [])

        # Determine if the record is Normal or Abnormal
        is_normal = all(event["type"] == "Normal" for event in event_annotations)
        record_bin_label = "Normal" if is_normal else "Abnormal"

        # Determine the abnormal types for the record
        abnormal_types = set(event["type"] for event in event_annotations if event["type"] != "Normal")
        record_abnormal_label = "NA" if is_normal else ", ".join(sorted(abnormal_types))

        # Store record summary for counting
        if record_name not in record_summary:
            record_summary[record_name] = {
                "record_bin_label": record_bin_label,
                "record_abnormal_label": record_abnormal_label
            }
        else:
            # Ensure consistency of record_bin_label across the same filename
            if record_summary[record_name]["record_bin_label"] == "Normal" and record_bin_label == "Abnormal":
                record_summary[record_name]["record_bin_label"] = "Abnormal"
                record_summary[record_name]["record_abnormal_label"] = record_abnormal_label
            elif record_summary[record_name]["record_bin_label"] == "Abnormal" and record_bin_label == "Abnormal":
                existing_abnormal_types = set(record_summary[record_name]["record_abnormal_label"].split(", "))
                updated_abnormal_types = existing_abnormal_types.union(abnormal_types)
                record_summary[record_name]["record_abnormal_label"] = ", ".join(sorted(updated_abnormal_types))

        # Process event-level information
        if not is_normal:
            for event in event_annotations:
                if event["type"] != "Normal":
                    onset = float(event["start"]) / 1000.0
                    offset = float(event["end"]) / 1000.0
                    event_label = event["type"]
                    data.append([record_name, onset, offset, event_label, record_bin_label, record_abnormal_label])
        else:
            onset = offset = event_label = "NA"
            data.append([record_name, onset, offset, event_label, record_bin_label, record_abnormal_label])

# Create a DataFrame using the collected data
df = pd.DataFrame(data,
                  columns=["filename", "onset", "offset", "event_label", "record_bin_label", "record_abnormal_label"])

# Save the DataFrame to a TSV file
df.to_csv(output_tsv_file, sep="\t", index=False)
print(f"Event data has been successfully saved to {output_tsv_file}")

# Count the number of samples of records contained in normal and abnormal at all record levels
record_counts = pd.DataFrame.from_dict(record_summary, orient='index')
record_level_counts = record_counts["record_bin_label"].value_counts()
print("\nNumber of samples in Normal and Abnormal records (Record Level):")
print(record_level_counts)
print(f"Total number of records: {len(record_counts)} (Should be equal to the number of JSON files: {len(json_files)})")

# Check for missing files
missing_files = set(json_files) - processed_files
if missing_files:
    print("\nMissing files that were not processed:")
    for missing_file in missing_files:
        print(missing_file)

# For all records containing abnormal, count the number of records of each abnormal type
abnormal_record_counts = record_counts[record_counts["record_bin_label"] == "Abnormal"][
    "record_abnormal_label"].value_counts()
print("\nNumber of records for each abnormal type (Record Level):")
print(abnormal_record_counts)
print(
    f"Total number of records with abnormal types: {abnormal_record_counts.sum()} (This number may exceed the number of records due to multiple abnormal types in one record)")

# For all records containing abnormal, count the number of each type of events
abnormal_event_records = df[df["record_bin_label"] == "Abnormal"]
abnormal_event_counts = abnormal_event_records["event_label"].value_counts()
print("\nNumber of each type of events in abnormal records (Event Level):")
print(abnormal_event_counts)
print(f"Total number of events in abnormal records: {abnormal_event_counts.sum()}")






"""
Event data has been successfully saved to train_set_2level_lab3.tsv

Number of samples in Normal and Abnormal records (Record Level):
Normal      1837
Abnormal     823
Name: record_bin_label, dtype: int64
Total number of records: 2660 (Should be equal to the number of JSON files: 2660)

Number of records for each abnormal type (Record Level):
Fine Crackle                            358
Wheeze                                  245
Fine Crackle, Wheeze                     92
Rhonchi                                  39
Coarse Crackle                           32
Stridor                                  18
Coarse Crackle, Fine Crackle              6
Fine Crackle, Wheeze+Crackle              6
Wheeze+Crackle                            5
Fine Crackle, Wheeze, Wheeze+Crackle      5
Coarse Crackle, Fine Crackle, Wheeze      4
Wheeze, Wheeze+Crackle                    3
Coarse Crackle, Wheeze                    3
Fine Crackle, Rhonchi                     2
Rhonchi, Wheeze                           1
Stridor, Wheeze                           1
Fine Crackle, Stridor                     1
Coarse Crackle, Rhonchi, Wheeze           1
Coarse Crackle, Rhonchi                   1
Name: record_abnormal_label, dtype: int64
Total number of records with abnormal types: 823 (This number may exceed the number of records due to multiple abnormal types in one record)

Number of each type of events in abnormal records (Event Level):
Fine Crackle      1204
Normal             878
Wheeze             871
Rhonchi            119
Coarse Crackle      85
Stridor             44
Wheeze+Crackle      32
Name: event_label, dtype: int64
Total number of events in abnormal records: 3233
"""





""" valid set

Number of samples in Normal and Abnormal records (Record Level):
Normal      487
Abnormal    177
Name: record_bin_label, dtype: int64
Total number of records: 664 (Should be equal to the number of JSON files: 664)

Number of records for each abnormal type (Record Level):
Fine Crackle                    76
Wheeze                          50
Fine Crackle, Wheeze            25
Rhonchi                         10
Coarse Crackle                   6
Stridor                          3
Wheeze+Crackle                   2
Coarse Crackle, Fine Crackle     2
Coarse Crackle, Rhonchi          1
Wheeze, Wheeze+Crackle           1
Rhonchi, Wheeze                  1
Name: record_abnormal_label, dtype: int64
Total number of records with abnormal types: 177 (This number may exceed the number of records due to multiple abnormal types in one record)

Number of each type of events in abnormal records (Event Level):
Fine Crackle      272
Wheeze            188
Normal            186
Rhonchi            29
Coarse Crackle     15
Stridor             5
Wheeze+Crackle      4
Name: event_label, dtype: int64
Total number of events in abnormal records: 699

"""









""""
Event data has been successfully saved to test_set_2level.tsv
Number of samples in Normal and Abnormal records (Record Level):
Normal      28
Abnormal    22
Name: record_bin_label, dtype: int64
Total number of records: 50 (Should be equal to the number of JSON files: 50)

Number of records for each abnormal type (Record Level):
Fine Crackle            9
Fine Crackle, Wheeze    9
Wheeze                  4
Name: record_abnormal_label, dtype: int64
Total number of records with abnormal types: 22 (This number may exceed the number of records due to multiple abnormal types in one record)

Number of each type of events in abnormal records (Event Level):
Fine Crackle    71
Wheeze          36
Normal          30
Name: event_label, dtype: int64
Total number of events in abnormal records: 137
"""