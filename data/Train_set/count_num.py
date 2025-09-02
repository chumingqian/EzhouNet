import pandas as pd

# Load the TSV file.
df = pd.read_csv('/media/respecting_god/S2/05_soundEventDetection/Respira_SED_LGNN/data/Train_set/train_set_refine_crackle.tsv', sep='\t')

# Display first few rows to check columns and values
print(df.head())

# Filter out rows where event_label is NaN or explicitly 'NA'
# Note: Depending on the file, missing values might be read as NaN automatically.
#abnormal_df = df[df['event_label'].notna() & (df['event_label'] != 'NA')]

# Filter out abnormal events, excluding NA (normal events) and compound events ("Wheeze+Crackle")
abnormal_df = df[
    df['event_label'].notna() &
    (df['event_label'] != 'NA') &
    (df['event_label'] != 'Wheeze+Crackle')  # Exclude compound events
]

# Count the number of each abnormal event type
event_counts = abnormal_df['event_label'].value_counts()
print("Event counts:\n", event_counts)

# Calculate proportions of each abnormal event type
total_abnormal = event_counts.sum()
event_proportions = event_counts / total_abnormal

print("\nEvent proportions:\n", event_proportions)




"""
[5 rows x 6 columns]
Event counts:
 Crackle           1289
Wheeze             871
Rhonchi            119
Stridor             44
Wheeze+Crackle      32
Name: event_label, dtype: int64

Event proportions:
 Crackle           0.547346
Wheeze            0.369851
Rhonchi           0.050531
Stridor           0.018684
Wheeze+Crackle    0.013588
Name: event_label, dtype: float64



Event counts:
 Crackle    1289
Wheeze      871
Rhonchi     119
Stridor      44
Name: event_label, dtype: int64

Event proportions:
 Crackle    0.554886
Wheeze     0.374946
Rhonchi    0.051227
Stridor    0.018941
Name: event_label, dtype: float64


"""