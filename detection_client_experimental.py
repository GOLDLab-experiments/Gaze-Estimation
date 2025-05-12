from socket import socket, AF_INET, SOCK_DGRAM
from inference import run_inference
import shutil
import os
import cv2

listen_ip = '0.0.0.0'
listen_port = 12345 

gaze_ip = '127.0.0.1'
gaze_port = 12346 

nimble_ip = '192.168.0.103'
nimble_port = 12347

s = socket(AF_INET, SOCK_DGRAM)
s.bind(("0.0.0.0", 12345))
print(f"Listening on {listen_ip}:{listen_port}") 

while True:
    if os.path.exists("camera_image.jpg"):
        os.remove("camera_image.jpg")

    # Send a message to the gaze detection server
    s.sendto(b'1', (gaze_ip, gaze_port))
    
    # Receive the looking_at_camera result from the gaze detection server
    gaze_data, _ = s.recvfrom(1)
    looking_at_camera = gaze_data.decode('utf-8')
    print(f"looking_at_camera: {looking_at_camera}")

    # Pass the image object to run_inference
    helmet = run_inference("camera_image.jpg", "itay")

    # cv2.imshow("Detection", cv2.imread("itay.py_helmet_dectections/predict/camera_image.jpg"))

    print(f"helmet: {helmet}")
    
    # Send the looking_at_camera result to the NIMBLE
    s.sendto(looking_at_camera.encode(), (nimble_ip, nimble_port))
    
    # Send the helmet result to the NIMBLE
    s.sendto(helmet.encode(), (nimble_ip, nimble_port))

    try:
        shutil.rmtree("itay.py_helmet_detections/predict")
    except Exception as e:
        print(f"Error removing directory: {e}")
        
s.close()
