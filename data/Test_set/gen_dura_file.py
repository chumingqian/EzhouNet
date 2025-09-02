import os
import pandas as pd
import torchaudio

# Folder path containing the .wav files
audio_folder = './test2024_detection_wav'  # <-- Replace with actual path

data = []

for file in os.listdir(audio_folder):
    if file.endswith('.wav'):
        filepath = os.path.join(audio_folder, file)
        try:
            waveform, sample_rate = torchaudio.load(filepath)
            duration = round(waveform.shape[1] / sample_rate, 3)  # duration in seconds
            data.append([file, duration])
        except Exception as e:
            print(f"Failed to load {file}: {e}")

# Create DataFrame
columns = ['filename', 'duration']
df = pd.DataFrame(data, columns=columns)

# Save to TSV
df.to_csv('test_audio_durations.tsv', sep='\t', index=False)
print("Saved audio durations to 'test_audio_durations.tsv'")
