import pandas as pd

# Read the CSV fil


# Load the newly uploaded TSV file and process it as per the user's request
file_path = "./valid_set_order.tsv"

# Read the TSV file
df = pd.read_csv(file_path, sep='\t')


def transform_csv(file_path):

    # Read the CSV data
    data = pd.read_csv(file_path, sep='\t')

    # Update event_label: redefine "Fine Crackle" and "Coarse Crackle" as "Crackle"
    data.loc[data['event_label'].isin(['Fine Crackle', 'Coarse Crackle']), 'event_label'] = 'Crackle'
    #data.loc[data['event_label'].isin(['Fine Crackle', 'Coarse Crackle']), 'record_abnormal_label'] = 'Crackle'

    # Replace empty values with 'NA'
    data = data.fillna('NA')

    # Save the modified DataFrame to a new CSV file
    data.to_csv('./valid_set_refine_crackle.tsv', index=False, sep='\t')


    return data





transformed_df = transform_csv(file_path)

# Print the transformed DataFrame
print(transformed_df)