# DriveCountAI

DriveCountAI is a cutting-edge computer vision project designed to monitor and analyze vehicular traffic flow in real-time. This project utilizes advanced computer vision algorithms, particularly object detection using the YOLO algorithm, to accurately count and track vehicles passing through specified areas.

## Features

- **Real-time Traffic Monitoring:** DriveCountAI provides real-time monitoring of vehicular traffic, offering instant insights into traffic patterns.
- **Accurate Vehicle Counting:** Leveraging the YOLO algorithm, it accurately counts and tracks vehicles, ensuring precise traffic analysis.
- **Configurable Parameters:** Users can customize settings and define specific areas for traffic monitoring, allowing flexibility in analysis.

## Requirements

- Python 3.x
- YOLO (You Only Look Once) algorithm libraries
- Camera / Live CCTV (Close Circuit TeleVision) footage / Video Input Source

## Installation

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Configure the input source (camera or CCTV or video feed) in the code.
4. Run the main script using `python drivecount.py`.

## Usage

1. Open the DriveCountAI python script.
2. Specify the designated areas for traffic monitoring by creating a mask (mask.png) in Canva or any preffered software.
3. Replace the designed mask file at the (mask) attribute. 
3. View real-time traffic counts and analysis on the dashboard.

## Contributing

Contributions and feedback are welcome! If you'd like to contribute to DriveCountAI, please fork the repository and create a pull request with your changes.

## Acknowledgments

DriveCountAI utilizes the YOLO algorithm, and we acknowledge the contributions and advancements made by the YOLO community in the field of object detection and computer vision.

Special Thanks to Murtaza for assisting me in building my pioneer project in the field of AI!
