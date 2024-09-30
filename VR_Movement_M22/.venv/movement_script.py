import pandas as pd
import numpy as np
import os
import csv
import warnings

def calculate_deltas_in_intervals(df, cols, interval=30):
    # the [name redacted] interval approach^tm
    # this script will calculate delta in 30 frame interval (1 second)
    # i do this instead of frame by frame as every student will move a similar speed frame by frame
    # but the entire second will better differentiate movements between students

    # initialization to ensure empty df per file
    deltas = []
    num_rows = len(df)

    # does the thing
    for i in range(0, num_rows, interval):
        end_index = min(i + interval, num_rows) - 1  # ensures non-30 frame intervals are covered
        if end_index > i:  # valid interval alert
            delta = df.loc[end_index, cols].values - df.loc[i, cols].values
            deltas.append(np.abs(delta))  # Store the absolute delta

    return np.array(deltas)  # np has dope functions


def process_file(file_path):
    # creates a file that stores deltas

    # extract participant ID from the file name
    participant_id = os.path.basename(file_path).split('.')[
        0]  # remember to rename files later

    # load the data
    data = pd.read_csv(file_path, delim_whitespace=True)

    # define columns for head and hand positions
    head_cols = ['HeadPosition_x', 'HeadPosition_y', 'HeadPosition_z']
    hand_cols = ['LeftHandPosition_x', 'LeftHandPosition_y', 'LeftHandPosition_z',
                 'RightHandPosition_x', 'RightHandPosition_y', 'RightHandPosition_z']

    # calculate the deltas in 30-frame intervals for the entire dataset
    head_deltas = calculate_deltas_in_intervals(data, head_cols)
    hand_deltas = calculate_deltas_in_intervals(data, hand_cols)

    # calculate the average of the interval deltas
    avg_head_delta = np.sum(head_deltas) ## my thought process behind sum rather than mean -- better differentiates bt active and non-active (non-vr) users
    #avg_head_delta = np.mean(np.sum(head_deltas)) # this is what using mean looks like
    avg_hand_delta = np.sum(hand_deltas)

    # prepare the output
    result_df = pd.DataFrame({
        'participant_id': [participant_id],
        'avg_head_delta': [avg_head_delta],
        'avg_hand_delta': [avg_hand_delta]
    })

    return result_df

# proving my insanity
def row_counter(file_path):
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        row_count = sum(1 for row in reader)  # Count the number of rows
    return row_count

def process_all_files_in_folder(folder_path):
    # iterates through all the files and starts the path through other functions

    all_results = pd.DataFrame()  # master file master file
    row_count = 0

    # does the thing
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # .tsv check
        if file_name.endswith('.tsv'):
            print(f"processing {file_path}")
            row_count = row_count + row_counter(file_path)
            result_df = process_file(file_path)
            all_results = pd.concat([all_results, result_df], ignore_index=True)

    # master file master file
    output_file = os.path.join(folder_path, 'all_participants_processed.csv')
    all_results.to_csv(output_file, index=False)
    print(f"All processed data saved to {output_file}")
    print(f"Row count: {row_count}")

# actual program that runs the things defined above
if __name__ == "__main__":

    # never replicate this line of code
    # do as i say not as i do
    warnings.filterwarnings("ignore", category=FutureWarning)
    # end of bad practices

    folder_path = './day_master'  # REPLACE WITH FILE PATH
    process_all_files_in_folder(folder_path) # starts the thing