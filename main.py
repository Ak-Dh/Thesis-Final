import cv2
import numpy as np
import dlib
import os
import argparse
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Set the environment variable to use XCB
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Load YOLO
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def process_video(video_path):
    # Open a video file
    cap = cv2.VideoCapture(video_path)

    # Get the screen size
    screen_width, screen_height = 2256, 1504  # Change these values to match your screen resolution

    # Create a resizable window
    cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Person Detection", screen_width, screen_height)
    # Set the desired frame interval
    frame_interval = 2  # Adjust this value as needed

    # Initialize a frame counter
    frame_counter = 1

    status_text = "No Face Detected"

    # Initialize gaze_values list
    gaze_values = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Increment the frame counter
        frame_counter += 1

        if frame_counter % frame_interval == 0:
            # Get the screen size
            screen_width, screen_height = 2256, 1504  # Change these values to match your screen resolution

            # Resize image to fit the screen
            frame = cv2.resize(frame, (screen_width, screen_height))

            # Detect objects using YOLO
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(layer_names)

            # Process detection results
            conf_threshold = 0.5

            # Lists to store bounding boxes, confidences, and class IDs
            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > conf_threshold and classes[class_id] == "person":
                        center_x = int(detection[0] * screen_width)
                        center_y = int(detection[1] * screen_height)
                        w = int(detection[2] * screen_width)
                        h = int(detection[3] * screen_height)

                        # Calculate bounding box coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Append to lists
                        boxes.append((x, y, w, h))
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to remove redundant overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

            # Display the image with bounding boxes
            for i in indices:
                i = i.item()  # Ensure i is an iterable
                box = boxes[i]
                (x, y, w, h) = box

                # Process each person individually
                roi = frame[y:y + h, x:x + w]

                # Check if the ROI is valid before face and gaze detection
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    # Convert the region of interest (ROI) to grayscale
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # Detect faces using Dlib
                    faces = detector(gray_roi)

                    # Check if the face is detected for this pedestrian
                    face_detected = len(faces) > 0

                    # Update pedestrian_status dictionary
                    # pedestrian_status[i] = face_detected

                    # Display face detection status above the bounding box
                    status_text = "Face Detected" if face_detected else "No Face Detected"
                    # cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Gaze detection logic
                    gaze_detected = False
                    if face_detected:
                        # Use Dlib's shape predictor to get facial landmarks
                        shape = predictor(roi, faces[0])

                        # Extract eye coordinates
                        left_eye_coords = shape.part(36).x, shape.part(36).y, shape.part(39).x, shape.part(39).y
                        right_eye_coords = shape.part(42).x, shape.part(42).y, shape.part(45).x, shape.part(45).y

                        # Draw rectangles around the eyes
                        cv2.rectangle(roi, (left_eye_coords[0], left_eye_coords[1]),
                                      (left_eye_coords[2], left_eye_coords[3]), (255, 0, 0), 2)
                        cv2.rectangle(roi, (right_eye_coords[0], right_eye_coords[1]),
                                      (right_eye_coords[2], right_eye_coords[3]), (255, 0, 0), 2)

                        # Set gaze_detected to True if eyes are detected
                        gaze_detected = True

                    # Append 0 or 1 to gaze_values based on gaze detection status
                    gaze_values.append(0 if gaze_detected else 1)

                    gaze_status_text = "Gaze Detected" if gaze_detected else "No Gaze Detected"
                    cv2.putText(frame, gaze_status_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Display the frame
            cv2.imshow("Person Detection", frame)
            cv2.waitKey(1)  # Add this line to play the video

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

    return gaze_values

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Detect persons and gaze in an image or video using YOLO and Dlib.')
parser.add_argument('--input', type=str, help='Path to the input file (video)', required=True)
args = parser.parse_args()

# Check if the input is an image or video
file_extension = os.path.splitext(args.input)[1].lower()

video_name = os.path.splitext(os.path.basename(args.input))[0]
print(f"Video Name: {video_name}")

# Process the video and get gaze_values
gaze_values = process_video(args.input)

# Load gaze values from video2.json
with open(f"annotation/{video_name}.json", "r") as file:
    data = json.load(file)
    video_id = data["video_id"]
    pedestrians = data["pedestrians"]

# Assuming there is only one pedestrian in the example JSON structure
pedestrian = pedestrians[0]
video_gaze_values = pedestrian["gaze"]

# Ensure the lengths match
min_length = min(len(gaze_values), len(video_gaze_values))
gaze_values = gaze_values[:min_length]
video_gaze_values = video_gaze_values[:min_length]

# Convert list to NumPy array for using sklearn metrics
gaze_array = np.array(gaze_values)
video_gaze_array = np.array(video_gaze_values)

# Initialize counters
false_positive = false_negative = true_positive = true_negative = 0

# Compare gaze values
for i in range(len(gaze_values)):
    if gaze_values[i] == 0 and video_gaze_values[i] == 1:
        false_positive += 1
    elif gaze_values[i] == 1 and video_gaze_values[i] == 0:
        false_negative += 1
    elif gaze_values[i] == 0 and video_gaze_values[i] == 0:
        true_positive += 1
    elif gaze_values[i] == 1 and video_gaze_values[i] == 1:
        true_negative += 1

# Calculate precision, recall, and f1_score
precision = precision_score(gaze_array, video_gaze_array)
recall = recall_score(gaze_array, video_gaze_array)
f1 = f1_score(gaze_array, video_gaze_array)

# Calculate accuracy
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

# Write results to fscore.txt
with open("fscore.txt", "a") as fscore_file:
    fscore_file.write(f"Video ID: {video_id}\n")
    fscore_file.write("Results:\n")
    fscore_file.write(f"False Positives: {false_positive}\n")
    fscore_file.write(f"False Negatives: {false_negative}\n")
    fscore_file.write(f"True Positives: {true_positive}\n")
    fscore_file.write(f"True Negatives: {true_negative}\n")
    fscore_file.write(f"Precision: {precision}\n")
    fscore_file.write(f"Recall: {recall}\n")
    fscore_file.write(f"F1 Score: {f1}\n")
    fscore_file.write(f"Accuracy: {accuracy}\n\n")

# Plot the metrics
plt.figure(figsize=(10, 8))  # Set overall figure size

# Plot Metrics Comparison
plt.subplot(3, 1, 1)
plt.bar(["False Positives", "False Negatives", "True Positives", "True Negatives"],
        [false_positive, false_negative, true_positive, true_negative], color=['red', 'orange', 'green', 'blue'])
plt.title('Metrics Comparison')
plt.xlabel('Metrics')
plt.ylabel('Count')

# Plot F1 Score
plt.subplot(3, 1, 2)
plt.bar(["F1 Score"], [f1], color=['purple'])
plt.title('F1 Score')
plt.xlabel('Metrics')
plt.ylabel('Value')

# Plot Accuracy
plt.subplot(3, 1, 3)
plt.bar(["Accuracy"], [accuracy], color=['cyan'])
plt.title('Accuracy')
plt.xlabel('Metrics')
plt.ylabel('Value')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
