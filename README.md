# Face Recognition Attendance System

## Overview
Developed a **Face Recognition Attendance System** to automate and streamline attendance tracking for organizations. This system uses real-time facial recognition to identify individuals and mark their attendance, improving both speed and accuracy while eliminating manual processes. Key features include:

1. **Face Encoding**: Pre-process images to store facial encodings of known individuals for future recognition.
2. **Real-Time Face Detection**: Detect and recognize faces in video streams from a webcam.
3. **Attendance Marking**: Automatically log attendance with the current timestamp if a recognized face is detected.
4. **CSV Attendance Records**: Store attendance data in a CSV file for easy reporting and management.
5. **Unknown Detection**: Handles unknown faces gracefully by labeling them as "Unknown."

## How I Developed the Application
Utilized the **power of Python** and **OpenCV** for real-time video capture, combined with **face_recognition** for facial detection and encoding. Key elements of the development process include:

- **Face Recognition Library**: Used to extract facial encodings and compare them with known faces.
- **OpenCV**: Handled video streaming and frame manipulation for a smooth experience.
- **NumPy**: Managed mathematical operations, such as calculating the closest match between encodings.
- **CSV Handling**: Automated attendance marking using Python’s **CSV library**.
- **Error Handling & Performance Optimization**: Enhanced error handling for missing faces and unknown encodings. Resized video frames to improve processing speed.

## Benefits to the Organization
1. **Automated Attendance Management**: Eliminated the need for manual attendance tracking, reducing human error and saving time.
2. **Improved Accuracy**: Reliably identifies individuals based on facial recognition, ensuring precise attendance logging.
3. **User-Friendly Interface**: Displays live video feed with real-time facial recognition results, making it easy for staff to monitor attendance.
4. **Data Export Capability**: Stores attendance data in CSV format, making it simple to review, analyze, and generate reports.
5. **Scalable for Future Use**: Easily extendable by adding new faces and supports different deployment environments.

## Challenges Faced
- **Low Lighting and Webcam Quality Issues**  
  **Solution**: Recommended optimal lighting conditions and webcam adjustments to users to ensure better recognition accuracy.

- **Handling Multiple Faces in Frame**  
  **Solution**: Implemented logic to process multiple faces independently, ensuring the system works in group settings.

- **Initial Setup Errors with Missing Face Encodings**  
  **Solution**: Added proper error handling to skip images without detectable faces and provided clear warning messages.
  
## Conclusion
The Face Recognition Attendance System provides a reliable and efficient way to automate attendance tracking in organizations. This project has enhanced my knowledge of computer vision and real-time video processing, and I look forward to applying these skills to future innovations.

## Full Code
```python
# IMPORTING NECESSARY LIBRARIES
import face_recognition
import cv2
import os
import glob
import numpy as np
import csv
from datetime import datetime

class FaceEncoder:
    """CLASS TO HANDLE FACE ENCODING AND DETECTION OF KNOWN FACES."""
    
    def __init__(self):
        # INITIALIZE VARIABLES TO STORE FACE ENCODINGS AND CORRESPONDING NAMES
        self.known_face_encodings = []
        self.known_face_names = []
        # FRAME RESIZING FACTOR FOR FASTER PROCESSING
        self.frame_resizing = 0.25  

    def load_encoding_images(self, images_path):
        """
        LOAD AND ENCODE ALL IMAGES FROM THE PROVIDED DIRECTORY.
        :param images_path: PATH TO THE FOLDER CONTAINING IMAGES FOR FACE RECOGNITION.
        """
        # GET ALL IMAGE FILES FROM THE DIRECTORY
        image_files = glob.glob(os.path.join(images_path, "*.*"))

        # CHECK IF IMAGES EXIST IN THE GIVEN DIRECTORY
        if not image_files:
            print("[ERROR] NO IMAGES FOUND. PLEASE PROVIDE IMAGES TO USE THIS TOOL.")
            return

        print(f"[INFO] ENCODING {len(image_files)} IMAGE{'S' if len(image_files) > 1 else ''}...")

        # ITERATE THROUGH EACH IMAGE AND EXTRACT FACE ENCODINGS
        for img_path in image_files:
            image = cv2.imread(img_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # GET IMAGE NAME WITHOUT EXTENSION
            filename = os.path.splitext(os.path.basename(img_path))[0]

            try:
                # EXTRACT FACE ENCODINGS FROM THE IMAGE (EXPECTING ONLY ONE FACE PER IMAGE)
                encoding = face_recognition.face_encodings(rgb_image)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(filename)
            except IndexError:
                print(f"[WARNING] NO FACE FOUND IN {img_path}. SKIPPING...")

        print(f"[INFO] LOADED AND ENCODED {len(self.known_face_encodings)} IMAGE{'S'}. TOOL IS READY.")

    def detect_known_faces(self, frame):
        """
        DETECT FACES IN THE GIVEN FRAME AND MATCH THEM WITH KNOWN ENCODINGS.
        :param frame: VIDEO FRAME TO PROCESS.
        :return: LIST OF FACE LOCATIONS AND CORRESPONDING NAMES.
        """
        # RESIZE THE FRAME TO IMPROVE PERFORMANCE
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # CONVERT BGR FRAME TO RGB (REQUIRED FOR FACE_RECOGNITION LIBRARY)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # FIND ALL FACE LOCATIONS AND ENCODINGS IN THE CURRENT FRAME
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        detected_names = []  # STORE NAMES OF DETECTED FACES

        for encoding in face_encodings:
            # COMPARE THE DETECTED FACE ENCODING WITH THE KNOWN ENCODINGS
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"  # DEFAULT NAME IF NO MATCH IS FOUND

            # FIND THE BEST MATCH BASED ON MINIMUM DISTANCE
            face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            detected_names.append(name)

        # SCALE BACK FACE LOCATIONS TO ORIGINAL FRAME SIZE
        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), detected_names

def get_csv_filename():
    """
    GENERATE A CSV FILENAME BASED ON TODAY'S DATE.
    :return: STRING FILENAME.
    """
    today_date = datetime.now().strftime("%d-%m-%Y")
    return f"attendance_{today_date}.csv"

def initialize_csv(filename):
    """
    CREATE A NEW CSV FILE WITH HEADERS IF IT DOESN'T ALREADY EXIST.
    :param filename: NAME OF THE CSV FILE.
    """
    if not os.path.isfile(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp"])
            print(f"[INFO] CREATED {filename} WITH HEADERS.")

def is_already_marked(name, filename):
    """
    CHECK IF THE GIVEN NAME IS ALREADY MARKED IN THE ATTENDANCE FILE.
    :param name: NAME TO CHECK.
    :param filename: CSV FILE NAME.
    :return: BOOLEAN VALUE (TRUE IF ALREADY MARKED, ELSE FALSE).
    """
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == name:
                    return True
    except FileNotFoundError:
        pass  # FILE NOT FOUND, RETURN FALSE
    return False

def mark_attendance(name, filename):
    """
    MARK ATTENDANCE FOR THE GIVEN NAME IF NOT ALREADY MARKED.
    :param name: NAME OF THE PERSON.
    :param filename: CSV FILE NAME.
    """
    if not is_already_marked(name, filename):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
            writer.writerow([name, timestamp])
            print(f"[INFO] MARKED ATTENDANCE FOR {name} AT {timestamp}.")

def main():
    """MAIN FUNCTION TO RUN THE ATTENDANCE SYSTEM."""
    # INITIALIZE CSV FILE FOR TODAY'S ATTENDANCE
    csv_filename = get_csv_filename()
    initialize_csv(csv_filename)

    # LOAD ENCODING IMAGES FROM THE SPECIFIED DIRECTORY
    face_encoder = FaceEncoder()
    face_encoder.load_encoding_images("images/")

    # INITIALIZE WEBCAM
    video_capture = cv2.VideoCapture(0)

    print("[INFO] STARTING VIDEO STREAM. PRESS 'ESC' TO EXIT.")

    while True:
        # CAPTURE FRAME FROM WEBCAM
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] FAILED TO CAPTURE FRAME. EXITING...")
            break

        # FLIP FRAME HORIZONTALLY FOR A MIRROR VIEW
        frame = cv2.flip(frame, 1)

        # DETECT FACES AND THEIR NAMES IN THE CURRENT FRAME
        face_locations, face_names = face_encoder.detect_known_faces(frame)

        # DRAW RECTANGLES AND NAMES AROUND DETECTED FACES
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # MARK ATTENDANCE IF THE PERSON IS RECOGNIZED
            if name != "Unknown":
                mark_attendance(name, csv_filename)

            # DRAW A RECTANGLE AROUND THE FACE
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # DISPLAY THE NAME BELOW THE FACE
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)

        # DISPLAY THE FRAME IN A WINDOW
        cv2.imshow("Attendance System", frame)

        # EXIT THE LOOP WHEN 'ESC' KEY IS PRESSED
        if cv2.waitKey(1) == 27:
            break

    # RELEASE VIDEO CAPTURE AND CLOSE ALL WINDOWS
    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] VIDEO STREAM STOPPED. PROGRAM TERMINATED.")

if __name__ == "__main__":
    main()
```

# License
This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for more details.
