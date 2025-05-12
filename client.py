from asyncio import sleep
import shutil
from socket import socket, AF_INET, SOCK_DGRAM
import time

import cv2
from matplotlib import pyplot as plt

from inference_helmet import run_inference_helmet
from skeleton import is_aggressive
 
s = socket(AF_INET, SOCK_DGRAM)
 
ip = '127.0.0.1'
port = 12346
 
while (True):
	s.sendto(b'1', (ip, port))
 
	data, sender_info = s.recvfrom (1)


	# Check for aggressive posture
	aggressive, person = is_aggressive("camera_image.jpg")

	# Pass the image object to run_inference
	helmet = run_inference_helmet("camera_image.jpg", "itay")
	
	
	processed_frame = cv2.imread("itay.py_helmet_detections/predict/camera_image.jpg")
	
	if helmet=='1':
		cv2.putText(processed_frame, "AUTHORIZED PERSONNEL", (15, 140), 
							cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


	if person=='1':
		cv2.putText(processed_frame, "Person detected!", (1000, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)		


	# cv2.imwrite("camera_image2.jpg", processed_frame)

	# Display
	cv2.imshow("Detection", processed_frame)
	cv2.waitKey(1)
	time.sleep(0.05)
	cv2.destroyAllWindows


	# Print the results
	print(f"gaze: {data.decode('utf-8')}")
	print(f"helmet: {helmet}")
	print(f"aggressive: {aggressive}")
	print(f"person: {person}")

        
	try:
		shutil.rmtree("itay.py_helmet_detections/predict")
	except Exception as e:
		print(f"Error removing directory: {e}")
 
s.close()