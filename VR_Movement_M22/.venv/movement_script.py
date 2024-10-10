import pandas as pd
import numpy as np
import os
import csv
import warnings

# setup instructions:
# pip install pandas numpy os csv warning

# fairly easy but probably not optimal
def calculate_distance_between_people(student_pos_x, student_pos_y, student_pos_z, prof_pos_x, prof_pos_y, prof_pos_z):

    # components of distance function
    x2 = pow(student_pos_x - prof_pos_x, 2)
    y2 = pow(student_pos_y - prof_pos_y, 2)
    z2 = pow(student_pos_z - prof_pos_z, 2)

    # calculates the distance
    distance = np.sqrt(x2 + y2 + z2)

    return distance

# This function should return a general idea of what direction the student is looking.
# refer to trigonometry textbooks, [name redacted] has textbook if need reference :)
def vector_from_rotation(rotation):
    # Assume rotation is a tuple (pitch, yaw) in degrees
    pitch, yaw = np.radians(rotation)

    # Calculate the forward vector from the head rotation (simplified version)
    forward_x = np.cos(pitch) * np.cos(yaw)
    forward_y = np.cos(pitch) * np.sin(yaw)
    forward_z = np.sin(pitch)

    return np.array([forward_x, forward_y, forward_z])

# general idea of this function is to figure out where the student is looking, and see if the professors coordinates are within
# that field of view.
# fairly barbaric calculation, as it doesn't necessarily account for distance
# function returns 1 for each line calculation, goal is to divide by 30 to get seconds looked at prof
def calculate_gaze_towards_instructor(student_pos, student_rotation, instructor_pos, threshold=0.9)
    print("Calculating Gaze")
    # Get the gaze direction of student
    student_gaze_direction = vector_from_rotation(student_rotation)

    # Calculate the direction from student to instructor
    direction_to_prof = np.array(instructor_pos) - np.array(student_pos)
    direction_to_prof = direction_to_prof / np.linalg.norm(direction_to_prof)  # Normalize

    # Calculate the dot product between the gaze direction and the direction to B
    dot_product = np.dot(student_gaze_direction, direction_to_prof)

    # Check if the dot product is above the threshold
    # threshhold is fairly arbitrary here, but general idea is there. probably will reduce to 70-80 if i can find some FoV lit
    if(dot_product > threshold):
        returnvalue = 1
    else:
        returnvalue = 0

    # general use would be to divide by 30 (30fps) to get a general seconds-looked-at-prof
    return returnvalue

# the [name redacted] interval approach^tm
# this script will calculate delta in 30 frame interval (1 second)
# i do this instead of frame by frame as every student will move a similar speed frame by frame
# but the entire second will better differentiate movements between students
def calculate_deltas_in_intervals(df, cols, interval=30):
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

# this function does all the delta calculations and file store.
def delta_process_file(file_path):
    # creates a file that stores deltas

    # extract participant ID from the file name
    participant_id = os.path.basename(file_path).split('.')[
        0]  # remember to rename files later

    # load the data
    data = pd.read_csv(file_path, delim_whitespace=True)

    # define columns for head and hand positions
    head_position = ['HeadPosition_x', 'HeadPosition_y', 'HeadPosition_z']
    head_rotation = ['HeadRotation_x', 'HeadRotation_y', 'HeadRotation_z']
    right_hand_position = ['RightHandPosition_x', 'RightHandPosition_y', 'RightHandPosition_z']
    left_hand_position = ['LeftHandPosition_x', 'LeftHandPosition_y', 'LeftHandPosition_z']
    right_hand_rotation = ['RightHandRotation_x', 'RightHandRotation_y', 'RightHandRotation_z']
    left_hand_rotation = ['LeftHandRotation_x', 'LeftHandRotation_y', 'LeftHandRotation_z']


    # calculate the deltas in 30-frame intervals for the entire dataset
    head_position_deltas = calculate_deltas_in_intervals(data, head_position)
    head_rotation_deltas = calculate_deltas_in_intervals(data, head_rotation)
    right_hand_position_deltas = calculate_deltas_in_intervals(data, right_hand_position)
    left_hand_position_deltas = calculate_deltas_in_intervals(data, left_hand_position)
    right_hand_rotation_deltas = calculate_deltas_in_intervals(data, right_hand_rotation)
    left_hand_rotation_deltas = calculate_deltas_in_intervals(data, left_hand_rotation)

    # calculate the average of the interval deltas
    avg_head_position_delta = np.sum(head_position_deltas) ## my thought process behind sum rather than mean -- better differentiates bt active and non-active (non-vr) users
    avg_head_rotation_delta = np.sum(head_rotation_deltas)

    #avg_head_delta = np.mean(np.sum(head_deltas)) # this is what using mean looks like
    avg_left_hand_position_delta = np.sum(left_hand_position_deltas)
    avg_right_hand_position_delta = np.sum(right_hand_position_deltas)
    avg_left_hand_rotation_delta = np.sum(left_hand_rotation_deltas)
    avg_right_hand_rotation_delta = np.sum(right_hand_rotation_deltas)

    # prepare the output
    result_df = pd.DataFrame({
        'participant_id': [participant_id],
        'avg_head_position': [avg_head_position_delta],
        'avg_head_rotation': [avg_head_rotation_delta],
        'avg_left_hand_position': [avg_left_hand_position_delta],
        'avg_right_hand_position': [avg_right_hand_position_delta],
        'avg_left_hand_rotation': [avg_left_hand_rotation_delta],
        'avg_right_hand_rotation': [avg_right_hand_rotation_delta]
    })

    return result_df

# proving my insanity
# not really needed for final computations. was just for descriptive
def row_counter(file_path):
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        row_count = sum(1 for row in reader)  # Count the number of rows
    return row_count

# this function goes through each folder in the directory and runs the functions
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
            result_df = delta_process_file(file_path) # DELTA CALCULATION, COMMENT OUT IF NOT RUNNING THAT DATA
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

    folder_path = './day_master'  # REPLACE WITH FILE PATH THIS JUST HAPPENS TO BE MY PATH IN THE VIRTUAL ENVIRONMENT
    process_all_files_in_folder(folder_path) # starts the thing