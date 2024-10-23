import pandas as pd
import numpy as np
import os
import csv
import warnings

gaze_towards_instructor = 0
folder_path = './master'  # REPLACE WITH FILE PATH THIS JUST HAPPENS TO BE MY PATH IN THE VIRTUAL ENVIRONMENT

# setup instructions:
# pip install pandas numpy os csv warning

# fairly easy but probably not optimal
def calculate_distance(student_pos_x, student_pos_y, student_pos_z, prof_pos_x, prof_pos_y, prof_pos_z):

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
def calculate_gaze_towards_instructor(student_pos, student_rotation, instructor_pos, threshold=0.9):
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
def calculate_deltas_in_intervals(df, cols):
    # initialization to ensure empty df per file
    deltas = []
    num_rows = len(df)

    # does the thing
    for i in range(0, num_rows -1):
        delta = df.loc[i+1, cols].values - df.loc[i, cols].values
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
    avg_left_hand_position_delta = np.mean(np.sum(left_hand_position_deltas))
    avg_right_hand_position_delta = np.mean(np.sum(right_hand_position_deltas))
    avg_left_hand_rotation_delta = np.mean(np.sum(left_hand_rotation_deltas))
    avg_right_hand_rotation_delta = np.sum(np.sum(right_hand_rotation_deltas))

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

def gaze_process_file(file_path, pPROF):
    # bringing global variable, c# habits die hard
    global gaze_towards_instructor

    # temp variables parsing file name
    participant_id = os.path.basename(file_path).split('.')[
        0]  # remember to rename files later
    class_no = os.path.basename(file_path).split('c')[
        1]  # remember to rename files later

    #print(class_no)

    correct_prof = [file for file in pPROF if class_no in file]
    global folder_path
    correct_prof = os.path.join(folder_path, correct_prof[0])

    prof_data = pd.read_csv(correct_prof, delim_whitespace=True)

    stu_data = pd.read_csv(file_path, delim_whitespace=True)
    #prof_data = pd.read_csv(correct_prof, delim_whitespace=True) # now checked by prveious if function
    prof_data.set_index('full_frame_no', inplace=True) # sets index for faster checking

    # define columns for head and hand positions
    frame_number = ['full_frame_no']
    head_position = ['HeadPosition_x', 'HeadPosition_y', 'HeadPosition_z']
    head_rotation = ['HeadRotation_x', 'HeadRotation_y', 'HeadRotation_z']

    distance_df = []

    rows_to_iterate = ['full_frame_no', 'HeadPosition_x', 'HeadPosition_y', 'HeadPosition_z', 'HeadRotation_x', 'HeadRotation_y', 'HeadRotation_z']
    for i, row in stu_data[rows_to_iterate].iterrows():
        row_number = row['full_frame_no']
        stu_pos = row['HeadPosition_x'], row['HeadPosition_y'], row['HeadPosition_z']
        stu_rot = row['HeadRotation_x'], row['HeadRotation_y'] # gaze only needs x, y, since roll (direction of z) can be inferred by sin(pitch)

        if row_number in prof_data.index:
            corresponding_row = prof_data.loc[row_number]
            ins_pos = corresponding_row['HeadPosition_x'], corresponding_row['HeadPosition_y'], corresponding_row[
                'HeadPosition_z']
            gaze_towards_instructor += calculate_gaze_towards_instructor(stu_pos, stu_rot, ins_pos)
            distance_df.append(calculate_distance(row['HeadPosition_x'], row['HeadPosition_y'], row['HeadPosition_z'], corresponding_row['HeadPosition_x'], corresponding_row['HeadPosition_y'], corresponding_row[
                'HeadPosition_z']))
        else:
            # print(f"    No match found for frame, skipping") # this was clogging the terminal, couldnt track by eye progress
            continue

    #length = len(file_path)
    student_total_time = (stu_data['full_frame_no'].iloc[-1] - stu_data['full_frame_no'].iloc[0]) / 30
    student_total_time = round(student_total_time, 2) # rounding to 2 for easy viewing
    gaze_towards_instructor = gaze_towards_instructor / 30
    gaze_towards_instructor = round(gaze_towards_instructor, 2)



    # prepare the output
    result_df = pd.DataFrame({
        'participant_id': [participant_id],
        'gazeins': [gaze_towards_instructor],
        'total_time': [student_total_time],
        'stuinsdis': [np.mean(distance_df)]
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
def delta_all_files_in_folder(folder_path):
    # iterates through all the files and starts the path through other functions

    all_results = pd.DataFrame()  # master file master file
    row_count = 0

    # does the thing
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # .tsv check
        if file_name.endswith('.tsv'):
            print(f"    processing file {file_path}")
            row_count = row_count + row_counter(file_path)
            result_df = delta_process_file(file_path) # DELTA CALCULATION, COMMENT OUT IF NOT RUNNING THAT DATA
            all_results = pd.concat([all_results, result_df], ignore_index=True)

    # master file master file
    output_file = os.path.join(folder_path, 'all_participants_delta.csv')
    all_results.to_csv(output_file, index=False)
    print(f"All processed data saved to {output_file}")
    print(f"Row count: {row_count}")

# process_all_files_in_folder, but for gaze not delta
def gaze_process_all_files(folder_path):
    # iterates through all the files and starts the path through other functions

    all_results = pd.DataFrame()  # master file master file
    row_count = 0

    p1 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    p8 = []
    p9 = []
    p10 = []
    p11 = []
    p12 = []
    p14 = []
    p15 = []
    p16 = []
    p17 = []
    p18 = []
    p19 = []
    p20 = []
    p21 = []
    p22 = []
    p24 = []
    p25 = []
    p26 = []
    p27 = []
    p28 = []
    p29 = []
    p30 = []
    p31 = []
    p34 = []
    p35 = []
    pPROF = []

    # does the thing
    for file_name in os.listdir(folder_path):

        # _ added to p1 and p3 to avoid adding files from p10, p30, etc
        if("p1_" in file_name):
            p1.append(file_name)
        if("p3_" in file_name):
            p3.append(file_name)
        if ("p4" in file_name):
            p4.append(file_name)
        if("p5" in file_name):
            p5.append(file_name)
        if("p6" in file_name):
            p6.append(file_name)
        if("p7" in file_name):
            p7.append(file_name)
        if("p8" in file_name):
            p8.append(file_name)
        if("p9" in file_name):
            p9.append(file_name)
        if("p10" in file_name):
            p10.append(file_name)
        if("p11" in file_name):
            p11.append(file_name)
        if("p12" in file_name):
            p12.append(file_name)
        if("p14" in file_name):
            p14.append(file_name)
        if("p15" in file_name):
            p15.append(file_name)
        if("p16" in file_name):
            p16.append(file_name)
        if ("p17" in file_name):
            p17.append(file_name)
        if("p18" in file_name):
            p18.append(file_name)
        if("p19" in file_name):
            p19.append(file_name)
        if("p20" in file_name):
            p20.append(file_name)
        if("p21" in file_name):
            p21.append(file_name)
        if("p22" in file_name):
            p22.append(file_name)
        if("p24" in file_name):
            p24.append(file_name)
        if("p25" in file_name):
            p25.append(file_name)
        if("p26" in file_name):
            p26.append(file_name)
        if("p27" in file_name):
            p27.append(file_name)
        if("p28" in file_name):
            p28.append(file_name)
        if("p29" in file_name):
            p29.append(file_name)
        if("p30" in file_name):
            p30.append(file_name)
        if("p31" in file_name):
            p31.append(file_name)
        if("p34" in file_name):
            p34.append(file_name)
        if("p35" in file_name):
            p35.append(file_name)
        if("pPROF" in file_name):
            pPROF.append(file_name)
    all_files = [p1, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p14, p15, p16, p17, p18, p19, p20, p21, p22, p24, p25, p26, p27, p28, p29, p30, p31, p34, p35]


    for file_name in all_files:
        for sub_file in file_name:
            # .tsv check
            if sub_file.endswith('.tsv'):
                print(f"    processing {sub_file}")
                file_path = os.path.join(folder_path, sub_file)
                with open(file_path, 'r') as f:
                    row_count = row_count + sum(1 for line in f) - 1 # -1 for headers
                result_df = gaze_process_file(file_path, pPROF)
                global gaze_towards_instructor
                gaze_towards_instructor = 0 # re-initialize
                all_results = pd.concat([all_results, result_df], ignore_index=True)

    # master file master file
    output_file = os.path.join(folder_path, 'all_participants_gaze.csv')
    all_results.to_csv(output_file, index=False)
    print(f"    All processed data saved to {output_file}")
    print(f"    Row count: {row_count}")


# actual program that runs the things defined above
if __name__ == "__main__":
    # never replicate this line of code
    # do as i say not as i do
    warnings.filterwarnings("ignore", category=FutureWarning)
    # end of bad practices

    #global folder_path
    print("PROCESSING DELTA STARTED")
    delta_all_files_in_folder(folder_path) # starts the delta calcs

    print("PROCESSING STU-INS GAZE & DISTANCE")
    gaze_process_all_files(folder_path) # starts both gaze and distance (at same time for optimization reasons)