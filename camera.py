# install opencv-python
import cv2
import typing
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                    "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                    "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                    "haarcascade_eye.xml")

def create_folders() -> None:
    if not os.path.exists("photos"):
        os.makedirs("photos")
    if not os.path.exists("videos"):
        os.makedirs("videos")

def detect_faces(frame) -> (bool, cv2.Mat):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # If faces are detected, it returns a numpy matrix where each row is a face (x, y, w, h), else it returns an empty matrix 
    faces_matrix = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

    # print("Detected faces:", faces_matrix)

    frame_with_faces = frame.copy()
    # For each row in the faces matrix
    for (x, y, w, h) in faces_matrix:
        frame_with_faces = cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=5)
        
    return frame_with_faces, faces_matrix

def snap_photo() -> (cv2.Mat, cv2.Mat):
    create_folders()

    stream = cv2.VideoCapture(1)

    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not stream.isOpened():
        print("No stream :(")
        exit()

    ret, frame = stream.read()
    frame_with_faces, faces_matrix = detect_faces(frame)
    
    if ret:
        cv2.imwrite("photos/snapshot.jpg", frame)
        # cv2.imshow("Snapshot", frame_with_face)
    if len(faces_matrix) > 0:
        cv2.imwrite("photos/snapshot_detected.jpg", frame_with_faces)
        # cv2.imshow("Snapshot", frame_with_face)
    # else:
    #     print("No faces detected")  # Debugging print statement
        
    stream.release()
    cv2.destroyAllWindows()

    return frame, faces_matrix