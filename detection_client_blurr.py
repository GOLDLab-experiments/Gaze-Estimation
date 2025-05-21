import shutil
import os
import sys
import time
import cv2
import numpy as np

from socket import socket, AF_INET, SOCK_DGRAM
from helmet import run_inference_helmet
from skeleton import is_aggressive, get_person_boundaries

gaze_ip = '127.0.0.1'
gaze_port = 12346
# ----------------------------------------------------

listen_ip = '0.0.0.0'
listen_port = 12345  

s = socket(AF_INET, SOCK_DGRAM)
s.bind(("0.0.0.0", 12345))
print(f"Listening on {listen_ip}:{listen_port}") 

# setup_fullscreen_window()

# Clean up any existing directories at startup
try:
    shutil.rmtree("itay.py_helmet_detections/predict")
except Exception as e:
    print(f"Error removing directory: {e}")

while True:
    # Clean up any existing images
    if os.path.exists("camera_image.jpg"):
        os.remove("camera_image.jpg")

    # Wait for connection from NIMBLE
    connection, (nimble_ip, nimble_port) = s.recvfrom(1)
    flush_image_buffer()  # Flush the camera buffer
    print(f"received call from nimble: {connection}")

    # If the NIMBLE wants to detect the face and helmet
    if connection:
        # Send a message to the gaze detection server
        s.sendto(b'1', (gaze_ip, gaze_port))
        
        # Receive the looking_at_camera result from the gaze detection server
        gaze_data, _ = s.recvfrom(1)
        looking_at_camera = gaze_data.decode('utf-8')

    # Check for aggressive posture
    aggressive, person = is_aggressive("camera_image.jpg")
    
    # Process the image based on person detection
    if person == '1':
        # Process the image to focus on the person area - this modifies camera_image.jpg in-place
        blur_except_person_area("camera_image.jpg", True)

        # Use the image for helmet detection
        helmet = run_inference_helmet("camera_image.jpg", "itay")
        result_path = "itay.py_helmet_detections/predict/camera_image.jpg"
        print(f"Helmet detection result: {helmet}") 
    
    else:
        helmet = '0'
        result_path = "camera_image.jpg"

    
    
    # Check if the processed image exists
    if os.path.exists(result_path):
        processed_frame = cv2.imread(result_path)
        
        # # Display
        # cv2.imshow("Detection", processed_frame)
        # key = cv2.waitKey(1)
        # if key == ord('q') or key == ord('Q'):
        #     print("Q key pressed, exiting...")
        #     cv2.destroyAllWindows()
        #     s.close()
        #     sys.exit(0)
        # time.sleep(0.05)

        # Print the results
        print(f"gaze: {gaze_data.decode('utf-8')}")
        print(f"helmet: {helmet}")
        print(f"aggressive: {aggressive}")
        print(f"person: {person}")

        # Send the results to the NIMBLE
        s.sendto(looking_at_camera.encode() +
                helmet.encode() + 
                aggressive.encode() +
                person.encode(),
                (nimble_ip, nimble_port))
    else:
        print(f"Error: Processed image not found at {result_path}")
        # Send default response
        s.sendto(looking_at_camera.encode() + "000".encode(), (nimble_ip, nimble_port))

    # Clean up directories
    try:
        shutil.rmtree("itay.py_helmet_detections/predict")
    except Exception as e:
        print(f"Error removing directory: {e}")

    connection = False
        
s.close()