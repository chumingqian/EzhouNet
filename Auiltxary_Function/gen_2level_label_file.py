import os
import json
import pandas as pd

# Define the folder containing the JSON files and the output file
# input_folder = "path/to/your/json/folder"
# output_tsv_file = "output_file.tsv"

# input_folder ="../data/Train_set/train_detection_json"
# output_tsv_file = "train_set.tsv"


# input_folder = "../data/Train_set/valid_detection_json"
# output_tsv_file = "valid_set.tsv"
#
#
# input_folder = "../data/Test_set"
# output_tsv_file = "test_set.tsv"
#
# # List to store the data extracted from all JSON files
import os
import json
import pandas as pd
import re

# Define the folder containing the JSON files and the output file
input_folder = "../data/Train_set/train_detection_json"
output_tsv_file = "train_set_2level.tsv"

# List to store the data extracted from all JSON files
data = []

# Iterate through all files in the specified folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(input_folder, filename)

        # Extract patient number and recording location from filename using regex
        match = re.match(r"(\d+)_([\d\.]+)_(\d+)_(p\d)_(\d+).json", filename)
        if match:
            patient_number = match.group(1)
            recording_location = match.group(4)
            record_name = filename.replace(".json", ".wav")

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

            # Process event-level information
            if is_normal:
                onset = offset = event_label = "NA"
                data.append([record_name, onset, offset, event_label, record_bin_label, record_abnormal_label])
            else:
                for event in event_annotations:
                    if event["type"] != "Normal":
                        onset = float(event["start"]) / 1000.0
                        offset = float(event["end"]) / 1000.0
                        event_label = event["type"]
                        data.append([record_name, onset, offset, event_label, record_bin_label, record_abnormal_label])

# Create a DataFrame using the collected data
df = pd.DataFrame(data,
                  columns=["filename", "onset", "offset", "event_label", "record_bin_label", "record_abnormal_label"])

# Save the DataFrame to a TSV file
df.to_csv(output_tsv_file, sep="\t", index=False)
print(f"Event data has been successfully saved to {output_tsv_file}")

# Count the number of samples of records contained in normal and abnormal at all record levels
record_counts = df["record_bin_label"].value_counts()
print("\nNumber of samples in Normal and Abnormal records:")
print(record_counts)

# For all records containing abnormal, count the number of records of each abnormal type
abnormal_records = df[df["record_bin_label"] == "Abnormal"]
abnormal_record_counts = abnormal_records["record_abnormal_label"].value_counts()
print("\nNumber of records for each abnormal type:")
print(abnormal_record_counts)

# For all records containing abnormal, count the number of each type of events
abnormal_event_counts = abnormal_records["event_label"].value_counts()
print("\nNumber of each type of events in abnormal records:")
print(abnormal_event_counts)

r"""
Data has been successfully saved to train_set.tsv

Sound Event Type Counts: 9785
Normal            7430
Fine Crackle      1204
Wheeze             871
Rhonchi            119
Coarse Crackle      85
Stridor             44
Wheeze+Crackle      32
Name: event_label, dtype: int64



Data has been successfully saved to valid_set.tsv

Sound Event Type Counts: 2428
Normal            1915
Fine Crackle       272
Wheeze             188
Rhonchi             29
Coarse Crackle      15
Stridor              5
Wheeze+Crackle       4
Name: event_label, dtype: int64


Data has been successfully saved to test_set.tsv

Sound Event Type Counts: 314
Normal          207
Fine Crackle     71
Wheeze           36
Name: event_label, dtype: int64


"""
import  wave
# input_folder = "../data/Train_set/train_detection_wav/"
# output_wav_tsv_file = "train_wav_druation.tsv"

#
# input_folder = "../data/Train_set/valid_detection_wav/"
# output_wav_tsv_file = "valid_wav_druation.tsv"
#
#
# input_folder = "./train_set.tsv"
# output_wav_tsv_file = "train_set_wav.tsv"


# input_folder = "./valid_set.tsv"
# output_wav_tsv_file = "valid_set_wav.tsv"


# input_folder = "./test_set.tsv"
# output_wav_tsv_file = "test_set_wav.tsv"
#
#
#
# import pandas as pd
#
#
# # Function to update filename suffixes in a TSV file
# def update_filename_suffix(tsv_file_path, updated_tsv_file_path):
#     # Read the TSV file
#     tsv_df = pd.read_csv(tsv_file_path, sep="\t")
#
#     # Replace all filename suffixes with .wav in the filename column
#     tsv_df["filename"] = tsv_df["filename"].str.replace(r"\.json$", ".wav", regex=True)
#
#     # Save the updated DataFrame to a new TSV file
#     tsv_df.to_csv(updated_tsv_file_path, sep="\t", index=False)
#     print(f"Updated TSV file has been successfully saved to {updated_tsv_file_path}")
#
#
# # Example usage of updating the filename suffix
# update_filename_suffix(input_folder, output_wav_tsv_file)