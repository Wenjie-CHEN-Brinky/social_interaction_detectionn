import os
from operator import itemgetter
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config

# Paths to config and checkpoint
config_path = 'C:/Users/Arthor/.cache/mim/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_path = 'C:/Users/Arthor/.cache/mim/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
label_map_path = '../mmaction2/tools/data/kinetics/label_map_behaviours.txt'
video_folder_path = '../yolov7-object-tracking/runs/detect/object_tracking3/output_clip_videos-test3'
output_txt_path = '../yolov7-object-tracking/runs/detect/object_tracking3/output_clip_videos-test3/behaviour_labels.txt'

# Initialize the recognizer
config = Config.fromfile(config_path)
model = init_recognizer(config, checkpoint_path, device='cuda:0')

# Read label map
with open(label_map_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Open the text file for writing results
with open(output_txt_path, 'w') as txt_file:
    # Loop over all video files in the folder
    for file_name in os.listdir(video_folder_path):
        if file_name.endswith('.mp4'):  # Only process .mp4 files
            video_path = os.path.join(video_folder_path, file_name)
            print(f'Processing video: {video_path}')

            # Inference to get behavior label
            results = inference_recognizer(model, video_path)
            pred_scores = results.pred_score.tolist()
            score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
            score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
            top_label_index, top_score = score_sorted[0]
            top_label = labels[top_label_index]

            # Create new file name with behavior label
            new_file_name = f"{file_name.split('.mp4')[0]}_{top_label}.mp4"
            new_file_path = os.path.join(video_folder_path, new_file_name)
            os.rename(video_path, new_file_path)  # Rename the video file

            # Write the result to the text file
            txt_file.write(f"{new_file_name} {top_label}\n")
            print(f"Renamed to: {new_file_name} | Label: {top_label} | Score: {top_score:.4f}")

print("Processing completed. Results saved to:", output_txt_path)