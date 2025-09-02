import os
import json
import pandas as pd
import wave
import re

#



# Define the folder containing the JSON files, WAV files, and the output files
input_folder ="../data/Train_set/train_detection_json"
output_tsv_file = "output_file.tsv"
output_wav_tsv_file = "output_wav_file.tsv"
output_abnormal_tsv_file = "output_abnormal_file.tsv"
output_patient_summary_tsv_file = "output2_patient_summary.tsv"



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

        # Open and load the JSON file
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

            # Iterate over event annotations in JSON data
            for event in json_data.get("event_annotation", []):
                # Extract start, end, and type fields and convert from ms to s
                onset = float(event.get("start", 0)) / 1000.0
                offset = float(event.get("end", 0)) / 1000.0
                event_label = event.get("type", "Unknown")

                # Append all events (including "Normal")
                data.append([filename, onset, offset, event_label, patient_number, recording_location])

# Create a DataFrame using the collected data
df = pd.DataFrame(data, columns=["filename", "onset", "offset", "event_label", "patient_number", "recording_location"])

# Save the DataFrame to a TSV file
df.to_csv(output_abnormal_tsv_file, sep="\t", index=False)

print(f"Event data has been successfully saved to {output_abnormal_tsv_file}")


# Function to count the number of each sound event type
def count_sound_event_types(df):
    event_counts = df["event_label"].value_counts()
    print("\nSound Event Type Counts:")
    print(event_counts)


# Count the number of each sound event type
count_sound_event_types(df)


# Function to count the number of times each abnormal event occurs in each position
def count_abnormal_events_by_position(df):
    abnormal_df = df[df["event_label"] != "Normal"]
    position_counts = abnormal_df.groupby(["event_label", "recording_location"]).size().unstack(fill_value=0)
    print("\nAbnormal Event Counts by Recording Location:")
    print(position_counts)


# Count the number of times each abnormal event occurs in each position
count_abnormal_events_by_position(df)


# Function to summarize patient information including abnormal events at each position and event type breakdown
def summarize_patient_info(df):
    patients = df["patient_number"].unique()
    summary_data = []

    for patient in patients:
        patient_df = df[df["patient_number"] == patient]
        abnormal_events = patient_df[patient_df["event_label"] != "Normal"]

        # Count abnormal events at each position (p1-p4)
        positions = ["p1", "p2", "p3", "p4"]
        position_counts = {position: "NA" for position in positions}
        for position in positions:
            count = abnormal_events[abnormal_events["recording_location"] == position].shape[0]
            if count > 0:
                position_counts[position] = count

        # Append patient summary data
        summary_data.append(
            [patient, position_counts["p1"], position_counts["p2"], position_counts["p3"], position_counts["p4"]])

        # Add rows for each specific type of anomaly at each location
        for event_label in abnormal_events["event_label"].unique():
            event_counts = []
            for position in positions:
                if position_counts[position] != "NA":
                    count = abnormal_events[(abnormal_events["recording_location"] == position) & (
                                abnormal_events["event_label"] == event_label)].shape[0]
                    event_counts.append(count if count > 0 else "NA")
                else:
                    event_counts.append("NA")
            summary_data.append(
                [f"{patient}_{event_label}", event_counts[0], event_counts[1], event_counts[2], event_counts[3]])

    # Create a DataFrame for patient summary
    summary_df = pd.DataFrame(summary_data, columns=["patient_number", "p1", "p2", "p3", "p4"])
    print("\nPatient Summary:")
    print(summary_df)

    # Save the patient summary to a TSV file
    summary_df.to_csv(output_patient_summary_tsv_file, sep="\t", index=False)
    print(f"Patient summary data has been successfully saved to {output_patient_summary_tsv_file}")


# Summarize patient information
summarize_patient_info(df)

# List to store data for WAV files
wav_data = []

# Iterate through all files in the specified folder to process WAV files
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        wav_path = os.path.join(input_folder, filename)

        # Open the WAV file to get duration
        with wave.open(wav_path, "r") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = round(frames / float(rate), 3)

            # Append the filename and duration to the list
            wav_data.append([filename, duration])

# Create a DataFrame using the collected WAV data
wav_df = pd.DataFrame(wav_data, columns=["filename", "duration"])

# Save the WAV DataFrame to a TSV file
wav_df.to_csv(output_wav_tsv_file, sep="\t", index=False)

print(f"WAV file data has been successfully saved to {output_wav_tsv_file}")