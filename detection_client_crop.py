import shutil
import os
import time
import cv2

from socket import socket, AF_INET, SOCK_DGRAM
from inference_helmet import run_inference_helmet
from skeleton import is_aggressive

gaze_ip = '127.0.0.1'
gaze_port = 12346

def flush_image_buffer():
    """
    Flush the camera buffer by capturing and discarding frames.
    """
    for _ in range(3):
        s.sendto(b'1', (gaze_ip, gaze_port))
        data, _ = s.recvfrom(1)

listen_ip = '0.0.0.0'
listen_port = 12345  

s = socket(AF_INET, SOCK_DGRAM)
s.bind(("0.0.0.0", 12345))
print(f"Listening on {listen_ip}:{listen_port}") 


try:
    shutil.rmtree("yolo11s.py_helmet_detections/predict")
except Exception as e:
    print(f"Error removing directory: {e}")


while True:


    if os.path.exists("camera_image.jpg"):
        os.remove("camera_image.jpg")

    connection, (nimble_ip, nimble_port) = s.recvfrom(64)
    # flush_image_buffer() # this is a hack to flush the camera buffer (im sorry no time :( )
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

	# Pass the image object to run_inference
    helmet = run_inference_helmet("camera_image.jpg", "yolo11s")
    
    processed_frame = cv2.imread("yolo11s.py_helmet_detections/predict/camera_image.jpg")

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
    # cv2.destroyAllWindows()


	# Print the results
    print(f"gaze: {gaze_data.decode('utf-8')}")
    print(f"helmet: {helmet}")
    print(f"aggressive: {aggressive}")
    print(f"person: {person}")

    # Send the looking_at_camera result to the NIMBLE
    s.sendto(looking_at_camera.encode() +
            helmet.encode() + 
            aggressive.encode() +
            person.encode(),
            (nimble_ip, nimble_port))

    # # Send the helmet result to the NIMBLE
    # s.sendto(helmet.encode(), (nimble_ip, nimble_port))

    # # Send the aggressive result to the NIMBLE
    # s.sendto(aggressive.encode(), (nimble_ip, nimble_port))

    # Send the aggressive result to the NIMBLE
    # s.sendto(aggressive.encode(), (nimble_ip, nimble_port))


    try:
        shutil.rmtree("yolo11s.py_helmet_detections/predict")
    except Exception as e:
        print(f"Error removing directory: {e}")

    connection = False
        
s.close()
