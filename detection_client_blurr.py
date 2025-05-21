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

def blur_except_person_area(image_path, person_detected):
    """
    Modifies the original image by blurring everything except the person and space above their head.
    Overwrites the original image and returns the same image path.
    """
    if not person_detected:
        return image_path
    
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return image_path
    
    try:
        # Get person boundaries using skeleton detection
        person_coords = get_person_boundaries(image_path)
        
        if person_coords is None:
            print("Warning: Could not detect person boundaries, using full image")
            return image_path
            
        # Unpack person coordinates
        x, y, w, h = person_coords
        
        # Add extra space above the head (approximately 30% of the person's height)
        extra_height = int(h * 0.3)
        y_start = max(0, y - extra_height)
        height = min(image.shape[0] - y_start, h + extra_height)
        
        # Create a blurred version of the entire image
        blurred_image = cv2.GaussianBlur(image, (55, 55), 0)
        
        # Create the final image: blurred background with clear person area
        result = blurred_image.copy()
        person_area = image[y_start:y_start+height, x:x+w]
        result[y_start:y_start+height, x:x+w] = person_area
        
        # Draw a rectangle around the person area (optional)
        # cv2.rectangle(result, (x, y_start), (x+w, y_start+height), (0, 255, 0), 2)
        
        # Save back to the original image path, overwriting it
        cv2.imwrite(image_path, result)
        
        return image_path
    except Exception as e:
        print(f"Error in blur_except_person_area: {e}")
        return image_path

def flush_image_buffer():
    """
    Flush the camera buffer by capturing and discarding frames.
    """
    for _ in range(3):
        s.sendto(b'1', (gaze_ip, gaze_port))
        data, _ = s.recvfrom(1)

# ----------------------------------------------------

listen_ip = '0.0.0.0'
listen_port = 12345  

s = socket(AF_INET, SOCK_DGRAM)
s.bind(("0.0.0.0", 12345))
print(f"Listening on {listen_ip}:{listen_port}") 

setup_fullscreen_window()

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
        
        # Write CS BIU on the image
        cv2.putText(processed_frame, "CS BIU", (10,55),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (40, 65, 0), 5)       

        if aggressive == '1':
            cv2.putText(processed_frame, "AGGRESSIVE BEHAVIOR!", (15, 100), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
            
        if helmet == '1':
            cv2.putText(processed_frame, "AUTHORIZED PERSONNEL", (15, 140), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
            
        if person == '1':
            cv2.putText(processed_frame, "Person detected!", (1000, 25), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

        # Display
        cv2.imshow("Detection", processed_frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            print("Q key pressed, exiting...")
            cv2.destroyAllWindows()
            s.close()
            sys.exit(0)
        time.sleep(0.05)

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