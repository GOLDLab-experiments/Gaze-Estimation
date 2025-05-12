from asyncio import sleep
import shutil
import os
from socket import socket, AF_INET, SOCK_DGRAM
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

from inference_helmet import run_inference_helmet
from skeleton import is_aggressive, get_person_boundaries

MODEL = 'itay'

# Function to blur everything except the person area
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
        cv2.rectangle(result, (x, y_start), (x+w, y_start+height), (0, 255, 0), 2)
        
        # Save back to the original image path, overwriting it
        cv2.imwrite(image_path, result)
        
        return image_path
    except Exception as e:
        print(f"Error in blur_except_person_area: {e}")
        return image_path

# Function to flush the camera buffer
def flush_camera_buffer(s, ip, port):
    """
    Flush the camera buffer by capturing and discarding frames.
    """
    print("Flushing camera buffer...")
    for _ in range(3):
        s.sendto(b'1', (ip, port))
        data, _ = s.recvfrom(1)
        time.sleep(0.1)
    print("Camera buffer flushed")

# Set up socket connection
s = socket(AF_INET, SOCK_DGRAM)
 
ip = '127.0.0.1'
port = 12346

# Flush the camera buffer at startup
# flush_camera_buffer(s, ip, port)

# Main loop
while True:
    # Clean up any existing image
    if os.path.exists("camera_image.jpg"):
        os.remove("camera_image.jpg")
    
    # Clean up any existing predictions directory
    try:
        shutil.rmtree(f"{MODEL}.py_helmet_detections/predict")
    except Exception as e:
        pass  # Directory might not exist yet
    
    # Send request for new image
    s.sendto(b'1', (ip, port))
    
    # Get the gaze detection result
    data, sender_info = s.recvfrom(1)
    looking_at_camera = data.decode('utf-8')
    
    # Wait for the image file to be available
    max_attempts = 20
    for attempt in range(max_attempts):
        if os.path.exists("camera_image.jpg"):
            # Found the file, give it a moment to finish writing
            time.sleep(0.2)
            print(f"Image found after {attempt+1} attempts")
            break
        time.sleep(0.1)
        print(f"Waiting for image... ({attempt+1}/{max_attempts})")
    
    if not os.path.exists("camera_image.jpg"):
        print("Error: camera_image.jpg not found")
        continue

    # Check for aggressive posture
    aggressive, person = is_aggressive("camera_image.jpg")
    
    # Save original image for comparison
    original_image = cv2.imread("camera_image.jpg")
    cv2.imwrite("original_image.jpg", original_image)

    # If a person is detected, blur everything except the person area
    if person == '1':
        # This modifies camera_image.jpg in-place
        blur_except_person_area("camera_image.jpg", True)
        print("Applied person-focused blurring")
    
    # Pass the processed image to run_inference
    helmet = run_inference_helmet("camera_image.jpg", MODEL)
    
    # Check if the processed image exists
    result_path = f"{MODEL}.py_helmet_detections/predict/camera_image.jpg"
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
        
        # Create a comparison view (optional)
        if os.path.exists("original_image.jpg"):
            original = cv2.imread("original_image.jpg")
            if original is not None and original.shape == processed_frame.shape:
                comparison = np.hstack((original, processed_frame))
                #cv2.imshow("Before/After Processing", comparison)
        
        # Display the processed image
        cv2.imshow("Detection", processed_frame)
        cv2.waitKey(1)
        time.sleep(0.05)
        
        # Print the results
        print(f"gaze: {looking_at_camera}")
        print(f"helmet: {helmet}")
        print(f"aggressive: {aggressive}")
        print(f"person: {person}")
    else:
        print(f"Error: Processed image not found at {result_path}")
    
    # Clean up predictions directory
    try:
        shutil.rmtree(f"{MODEL}.py_helmet_detections/predict")
    except Exception as e:
        print(f"Error removing directory: {e}")
    
    # Short delay before next iteration
    time.sleep(0.5)
 
s.close()