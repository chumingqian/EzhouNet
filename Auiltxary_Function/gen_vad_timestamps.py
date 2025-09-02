# import torch
# torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint


from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
# model = load_silero_vad()
# wav = read_audio('./en.wav')
#
# speech_timestamps = get_speech_timestamps(
#   wav,
#   model,
#   return_seconds=True,  # Return speech timestamps in seconds (default is samples)
# )
#
#
# pprint(speech_timestamps)
#


import os
import torch
import concurrent.futures
import pandas as pd
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from tqdm import tqdm

# Set the number of threads to avoid overloading the system
torch.set_num_threads(1)

# Load the VAD model
model = load_silero_vad()

# Function to process a single audio file
audio_folder = "../data/Train_set/valid_detection_wav"


import os
import torch
import multiprocessing
import pandas as pd
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from tqdm import tqdm

# Set the number of threads to avoid overloading the system
torch.set_num_threads(1)

# Load the VAD model
model = load_silero_vad()

# Function to process a single audio file
# Function to process a single audio file
def process_audio(file_path):
    try:
        wav = read_audio(file_path)
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
        if speech_timestamps:
            onset = speech_timestamps[0]['start']
            offset = speech_timestamps[-1]['end']
        else:
            onset = 'NA'
            offset = 'NA'
        return [os.path.basename(file_path), onset, offset]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [os.path.basename(file_path), 'NA', 'NA']

# Folder containing the audio files
audio_folder = "../data/Train_set/valid_detection_wav" # Replace with the path to your folder containing .wav files

# Get a list of all .wav files in the folder
audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]

# Process files in parallel using multiprocessing and collect results
with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
    results = list(tqdm(pool.imap(process_audio, audio_files), total=len(audio_files)))

# Create a DataFrame and save the results to a .tsv file
columns = ["filename", "onset", "offset"]
df = pd.DataFrame(results, columns=columns)
df.to_csv('speech_timestamps.tsv', sep='\t', index=False)

# Count the number of detection events and calculate the ratio
detected_count = df[(df['onset'] != 'NA') & (df['offset'] != 'NA')].shape[0]
total_count = df.shape[0]
detection_ratio = detected_count / total_count

print("Results saved to speech_timestamps.tsv")
print(df)
print(f"Number of detection events: {detected_count}")
print(f"Detection ratio: {detection_ratio:.2f}")

