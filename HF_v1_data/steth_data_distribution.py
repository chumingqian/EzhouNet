import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define event types (including Wheeze+Crackle as a distinct type)
EVENT_TYPES = ['Crackle', 'Wheeze', 'Stridor', 'Rhonchi', 'Wheeze+Crackle']

# Define duration intervals
DURATION_BINS = [0, 0.5] + [0.5 + 0.25 * i for i in range(1, 11)]  # [0, 0.5, 0.75, 1.0, ..., 3.0]
DURATION_LABELS = ['0-0.5s'] + [f'{DURATION_BINS[i]:.2f}-{DURATION_BINS[i + 1]:.2f}s' for i in
                                range(1, len(DURATION_BINS) - 1)]


def read_and_process_tsv(tsv_path="train_set_refine_crackle.tsv"):
    # Read TSV file
    df = pd.read_csv(tsv_path, sep='\t')

    # Filter for filenames starting with 'steth' (exclude 'trunc')
    df = df[df['filename'].str.startswith('steth')].copy()

    # Filter for relevant event types
    df = df[df['event_label'].isin(EVENT_TYPES)].copy()

    # Convert onset and offset to numeric, handling NA
    df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
    df['offset'] = pd.to_numeric(df['offset'], errors='coerce')

    # Calculate duration
    df['duration'] = df['offset'] - df['onset']

    # Remove invalid durations (e.g., negative, zero, or NaN)
    df = df[df['duration'] > 0].dropna(subset=['duration'])

    return df


def compute_statistics(df):
    # Count total number of each event type
    event_counts = df['event_label'].value_counts().reindex(EVENT_TYPES, fill_value=0)

    # Initialize dictionary to store binned counts
    binned_counts = {event: np.zeros(len(DURATION_BINS) - 1) for event in EVENT_TYPES}

    # Bin durations for each event type
    for event in EVENT_TYPES:
        event_df = df[df['event_label'] == event]
        if not event_df.empty:
            # Histogram of durations
            counts, _ = np.histogram(event_df['duration'], bins=DURATION_BINS)
            binned_counts[event] = counts

    # Convert to DataFrame
    binned_df = pd.DataFrame(binned_counts, index=DURATION_LABELS)

    # Calculate proportions (normalize by total count of each event type)
    proportions_df = binned_df.div(event_counts, axis=1).fillna(0)

    return event_counts, binned_df, proportions_df


def plot_histogram(proportions_df, data_stage):
    # Set up the plot
    plt.figure(figsize=(14, 8))

    # Plot settings
    bar_width = 0.15  # Adjusted for 5 event types
    x = np.arange(len(DURATION_LABELS))

    # Plot bars for each event type
    for i, event in enumerate(EVENT_TYPES):
        plt.bar(x + i * bar_width, proportions_df[event], bar_width, label=event, alpha=0.8)

    # Customize plot
    plt.xlabel('Duration Intervals')
    plt.ylabel('Proportion of Events')
    plt.title(f'Proportion of Event Types on {data_stage} by Duration Interval')
    plt.xticks(x + bar_width * (len(EVENT_TYPES) - 1) / 2, DURATION_LABELS, rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_path = data_stage + '_event_duration_steth_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    # Read and process TSV
    # path = './train_annotations.tsv'
    # data_stage = 'HF_v1_Train'

    path = './valid_annotations.tsv'
    data_stage = 'HF_v1_valid'
    df = read_and_process_tsv(tsv_path=path)

    # Compute statistics
    event_counts, binned_counts, proportions_df = compute_statistics(df)

    # Print statistics
    print("\nTotal Number of Each Event Type:")
    print(event_counts)
    print("\nCount of Events in Each Duration Interval:")
    print(binned_counts)
    print("\nProportion of Events in Each Duration Interval:")
    print(proportions_df)

    # Plot histogram
    plot_path = plot_histogram(proportions_df, data_stage)
    print(f"\nHistogram saved to: {plot_path}")


if __name__ == "__main__":
    main()