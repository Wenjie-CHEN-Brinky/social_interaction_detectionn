import os
import numpy as np
import cv2
from collections import defaultdict

# Define constants
DISTANCE_THRESHOLD = 3.7  # meter
SQUARE_SIDE_LENGTH = 0.15  # Length of the square's side (in normalized coordinates)
CONSECUTIVE_FRAMES_THRESHOLD = 300

# Coordinates of the points in the real world
real_world_points = np.array([
    [458385.9, 304497.6],
    [458373.9, 304499.4],
    [458378.9, 304473.8],
    [458396.81, 304454.86],
    [458407.6, 304459.4],
    [458353.7, 304433.7]
], dtype=np.float32)

# Coordinates of the points in the video (normalized coordinates)
video_points = np.array([
    [0.1679, 0.8013],
    [0.6203, 0.9454],
    [0.5460, 0.4847],
    [0.3105, 0.3331],
    [0.1300, 0.3346],
    [0.8911, 0.3211]
], dtype=np.float32)

# Compute the homography matrix
H, _ = cv2.findHomography(video_points, real_world_points)

# Function to transform video points to real-world coordinates
def transform_to_real_world(video_point, H):
    video_point_h = np.array([video_point[0], video_point[1], 1.0])  # Homogeneous coordinates
    real_world_point_h = np.dot(H, video_point_h)
    real_world_point = real_world_point_h[:2] / real_world_point_h[2]  # Convert back to Cartesian coordinates
    return real_world_point

# Function to calculate Euclidean distance
def calculate_distance(coord1, coord2):
    real_world_coord1 = transform_to_real_world(coord1, H)
    real_world_coord2 = transform_to_real_world(coord2, H)
    return np.sqrt((real_world_coord1[0] - real_world_coord2[0])**2 + (real_world_coord1[1] - real_world_coord2[1])**2)

# Parse TXT files
def parse_tracking_data(folder_path):
    data = defaultdict(list)  # {frame_id: [(object_id, x, y)]}
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.txt'):
            frame_id = int(file_name.split('_')[-1].split('.')[0])  # Extract frame ID from file name
            with open(os.path.join(folder_path, file_name), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    object_id = int(parts[0]) + 1
                    x, y = float(parts[2]), float(parts[3])
                    data[frame_id].append((object_id, x, y))
    print("Parse Tracking Data...Finished")
    return data

# Identify close pairs
def find_close_pairs(data):
    close_pairs = defaultdict(list)  # {frame_id: [(id1, id2)]}
    for frame_id, objects in data.items():
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i < j:
                    distance = calculate_distance(obj1[1:], obj2[1:])
                    if distance < DISTANCE_THRESHOLD:
                        close_pairs[frame_id].append((obj1[0], obj2[0], distance))
    print("Find Close Pairs...Finished")
    return close_pairs

# Track pairs across consecutive frames
def track_pairs(close_pairs):
    pair_durations = defaultdict(list)  # {(id1, id2): [(start_frame, end_frame)]}
    active_pairs = {}  # {(id1, id2): start_frame}

    for frame_id in sorted(close_pairs.keys()):
        for pair in close_pairs[frame_id]:
            ids = pair[:2]
            if ids in active_pairs:
                continue
            else:
                active_pairs[ids] = frame_id

        inactive_pairs = [ids for ids in active_pairs if ids not in [p[:2] for p in close_pairs[frame_id]]]
        for ids in inactive_pairs:
            start_frame = active_pairs.pop(ids)
            pair_durations[ids].append((start_frame, frame_id - 1))

    for ids, start_frame in active_pairs.items():
        pair_durations[ids].append((start_frame, max(close_pairs.keys())))

    print("Track Pairs...Finished")
    return pair_durations

# Filter pairs by duration
def filter_long_pairs(pair_durations):
    long_pairs = []
    for pair, durations in pair_durations.items():
        for start, end in durations:
            if end - start + 1 >= CONSECUTIVE_FRAMES_THRESHOLD:
                long_pairs.append((pair, start, end))
    print("Filter Long Pairs...Finished")
    return long_pairs

# Extract videos and save details to a TXT file
def extract_videos_with_details(video_path, long_pairs, data, close_pairs, output_folder, txt_output_path):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with open(txt_output_path, 'w') as txt_file:
        txt_file.write("Start_Frame,End_Frame,Duration(s),ID_Count,IDs,Positions,Distances\n")

        for idx, (pair, start_frame, end_frame) in enumerate(long_pairs):
            id1, id2 = pair
            duration = (end_frame - start_frame + 1) / fps

            ids_in_range = set()
            positions = []
            distances = []

            for frame_id in range(start_frame, end_frame + 1):
                if frame_id in close_pairs:
                    for close_pair in close_pairs[frame_id]:
                        ids_in_range.update(close_pair[:2])
                        positions.append(f"Frame {frame_id}: {close_pair[:2]} -> {close_pair[2]:.2f}m")
                        distances.append(close_pair[2])

            video_name = os.path.join(
                output_folder,
                f"situation_{idx+1}_IDs_{id1}_{id2}_frames_{start_frame}_{end_frame}.mp4"
            )
            out = cv2.VideoWriter(
                video_name,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_id in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()

            txt_file.write(
                f"{start_frame},{end_frame},{duration:.2f},{len(ids_in_range)},{list(ids_in_range)},\"{' | '.join(positions)}\",\"{distances}\"\n"
            )

    cap.release()
    print("Videos and details saved.")

# Main process
folder_path = "../processing/runs/detect/object_tracking4/labels"
video_path = "../processing/runs/detect/object_tracking4/3.mp4"
output_folder = "../processing/runs/detect/object_tracking4/distance_2"
txt_output_path = "../processing/runs/detect/object_tracking4/distance_2/details.txt"

data = parse_tracking_data(folder_path)
close_pairs = find_close_pairs(data)
pair_durations = track_pairs(close_pairs)
long_pairs = filter_long_pairs(pair_durations)
extract_videos_with_details(video_path, long_pairs, data, close_pairs, output_folder, txt_output_path)

