# Pedestrian Gaze Detection Project

This project implements pedestrian detection in images and videos using YOLO (You Only Look Once) and gaze detection using Dlib.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Scripts](#scripts)
- [Dependencies](#dependencies)

## Prerequisites

- Python 3.6 or higher
- Virtual environment (optional but recommended)
- Download the JAAD annotations and JAAD Clips from https://github.com/ykotseruba/JAAD
- Download YOLO weights from https://pjreddie.com/media/files/yolov3.weights

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/Ak-Dh/Thesis.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Thesis
    ```

3. (Optional) Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4. Make sure to place the videos into data/vid folder

## Usage

1. Run the `main.py` script to detect pedestrians and their gaze in any given input:

    The input must be like this:
    ```bash
    python main.py ./data/vid/[video_name].mp4
    ```

## Scripts

- `main.py`: Detects pedestrians and their gaze and determins the accuracy with which the framework prdicts the output and the f1 score with the provided input
- `generate_images.py`: Dused to generate the frames from the video with frame interval 2.
- `jaad_data.py`: Has list of JAAD dataset fucntions.
- `model.py`: Python script that executes the framework with JAAD video clips and JAAD annotations
- `setup_jaad.py`: Used to intialise the JAAD dataset

## Dependencies

- OpenCV: For image and video processing.

    ```bash
    pip install opencv-python
    ```

- Dlib: For gaze detection.

    ```bash
    pip install dlib
    ```

**Note**: Make sure to replace  `[video_name].mp4` with the actual names of your image and video files.

