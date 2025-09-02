# Load the TSV file and inspect its content
import pandas as pd

# Load the TSV file
# file_path = "./valid_annotations.tsv"
file_path = "./train_annotations.tsv"
df = pd.read_csv(file_path, sep='\t')

# Display the first few rows to understand the structure
# df.head()
#


# Remove the "_label" suffix from each filename
df["filename"] = df["filename"].str.replace("_label.wav", ".wav", regex=False)

# Save the modified DataFrame to a new TSV file
new_file_path = "./train_annotations_modified.tsv"
df.to_csv(new_file_path, sep='\t', index=False)

new_file_path
