import cv2
import numpy as np
from jaad_data import JAAD
import os

# Set the environment variable to use XCB
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Load YOLOv3 model and configuration
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize JAAD database
jaad_path = "./Jaad"
imdb = JAAD(data_path=jaad_path)

# Choose a video from JAAD_Clips
video_path = "./Jaad/JAAD_clips/video_0001.mp4"  # Change this to the desired video path
cap = cv2.VideoCapture(video_path)

# Get video dimensions
width = int(cap.get(3))
height = int(cap.get(4))

# Open the output file
output_file = open("output.txt", "w")

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass to get detections
    output_layer_name = "yolo_106"  # Replace with the actual output layer name for your YOLOv3 model
    detections = net.forward(output_layer_name)

    # Loop through detections
    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter detections based on confidence threshold
        if confidence > 0.5 and class_id == classes.index("person"):
            # Get bounding box coordinates
            box = detection[0:4] * np.array([width, height, width, height])
            (x, y, w, h) = box.astype("int")

            # Get JAAD annotations for the pedestrian
            video_id = "video_0001"  # Change this to the actual video ID
            annotations = imdb._get_annotations(video_id)

            # Display the annotations for each pedestrian
            for ped_id, ped_data in annotations['ped_annotations'].items():
                # Save to the output file
                output_file.write(f"{video_id} Pedestrian ID:{ped_id} INFO:{ped_data['behavior']}\n")

    # Resize image to fit the screen
    frame = cv2.resize(frame, (width, height))

    # Create a named window and set its size
    cv2.namedWindow("YOLOv3 Pedestrian Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv3 Pedestrian Detection", width, height)

    # Display the frame with detections
    cv2.imshow("YOLOv3 Pedestrian Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, close the output file, and close the OpenCV window
cap.release()
output_file.close()
cv2.destroyAllWindows()
