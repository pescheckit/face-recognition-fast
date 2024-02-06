# Advanced Face Recognition System

## Introduction
This Python script, `poc.py`, utilizes OpenCV and the `face_recognition` library to detect and compare faces in two different images. It efficiently handles varying orientations and provides a similarity score, making it ideal for applications in identity verification and security systems.

## Prerequisites
- Python 3.6 or higher
- OpenCV
- face_recognition
- NumPy

## Installation
1. Clone the repository:
   ```
   git clone [Your Repository URL]
   ```
2. Navigate to the cloned directory:
   ```
   cd [Your Repository Directory]
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the script by passing two image paths as arguments:
```
python poc.py path_to_id_image path_to_webcam_image
```
- `path_to_id_image`: Path to the ID or reference image.
- `path_to_webcam_image`: Path to the webcam or test image.

## Features
- Detects and extracts faces in different orientations.
- Compares faces to determine a match and calculates a similarity percentage.
- Handles rotated or poorly aligned images.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
