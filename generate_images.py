import cv2
import os

def generate_images_for_all_videos(video_folder, output_folder, frame_interval=2):
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # List all video files in the video folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # Extract video name without extension

        # Create a subfolder within "images" for each video
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Iterate through frames with the specified interval and save as images
        for frame_num in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                image_path = os.path.join(video_output_folder, f"frame_{frame_num}.png")
                cv2.imwrite(image_path, frame)

        cap.release()

# Example Usage:
video_folder = "data/vid"
output_folder = "images"

generate_images_for_all_videos(video_folder, output_folder, frame_interval=2)
