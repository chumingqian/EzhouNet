import pandas as pd

# Load the .tsv file into a DataFrame
df = pd.read_csv('../../Auiltxary_Function/test_set_wav.tsv', sep='\t')

# Sort the DataFrame by 'filename' and 'onset' columns
df_sorted = df.sort_values(by=['filename', 'onset'])

# Save the sorted DataFrame back to a .tsv file
df_sorted.to_csv('test_set_order.tsv', sep='\t', index=False)

print("File sorted and saved as 'train_set_order.tsv'.")