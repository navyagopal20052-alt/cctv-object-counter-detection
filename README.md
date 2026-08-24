# CCTV Object Counter Detection

An AI-based CCTV emergency object detection and counting system using YOLOv8, Python, OpenCV, Flask, HTML, CSS, and JavaScript.

## Project Overview

This project analyzes CCTV/video input and uses a trained YOLOv8 model to detect and count emergency-related objects or events. The system provides a web-based interface for viewing detection results.

## Features

* Emergency object/event detection from CCTV/video input
* Object counting
* YOLOv8-based detection
* Flask backend
* Web-based frontend
* Alert sound for detected emergencies
* Detection output generation and download
* Testing utilities for the detection model

## Technologies Used

* Python
* YOLOv8
* OpenCV
* Flask
* HTML
* CSS
* JavaScript

## How It Works

1. CCTV/video input is provided to the system.
2. The YOLOv8 model processes the input.
3. The system detects trained emergency categories.
4. Detected objects/events are counted.
5. The result is displayed through the web interface.
6. An alert can be generated for detected emergencies.

## Project Structure

```text
cctv-object-counter-detection/
├── app.py
├── requirements.txt
├── create_labels.py
├── create_alert_bell.py
├── test_models.py
├── index.html
└── alert-bell.wav
```

## Limitations

Detection performance depends on the trained model, training data, image quality, lighting conditions, camera angle, and type of input provided. The system may not detect every object or situation accurately.

## Future Improvements

* Improve model accuracy with a larger and more diverse dataset
* Add real-time CCTV camera streaming
* Improve emergency classification
* Add a database for storing detection history
* Add an analytics dashboard
* Deploy the application as a live web application

## Note

The trained YOLOv8 model file and original dataset are not included in this repository because of file size and repository management considerations.

## Screenshots

![Project Interface](Screenshot%20%28156%29.png)

![Detection Result](Screenshot%20%28161%29.png)

![Detection Output](Screenshot%20%28163%29.png)
