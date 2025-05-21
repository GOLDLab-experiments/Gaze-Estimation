import time
import cv2
import mediapipe as mp
import numpy as np
import skeleton

# From detection client
def setup_fullscreen_window():
    """
    Create a named window in fullscreen mode
    """
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def blur_except_person_area(image, person_detected):
    """
    Modifies the original image by blurring everything except the person and space above their head.
    Overwrites the original image and returns the same image path.
    """
    if not person_detected:
        return None
    
    # Load the original image
    if image is None:
        print(f"Error: Could not read image ")
        return None
    
    try:
        # Get person boundaries using skeleton detection
        person_coords = skeleton.get_person_boundaries(image)
        
        if person_coords is None:
            print("Warning: Could not detect person boundaries, using full image")
            return None
            
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
        
        
        return image
    except Exception as e:
        print(f"Error in blur_except_person_area: {e}")
        return image


def decorate_the_image(image, aggressive, helmet, person):

    
    # Write CS BIU on the image
    cv2.putText(image, "CS BIU", (10,55),
                cv2.FONT_HERSHEY_TRIPLEX, 2, (40, 65, 0), 5)       

    # Write aggressive behavior on the image
    if aggressive:
        cv2.putText(image, "AGGRESSIVE BEHAVIOR!", (15, 100), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

    # Write person detected on the image
    if person:
        cv2.putText(image, "Person detected!", (1000, 25), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
   
    # Write authorized personnel on the image
    if helmet: 
        cv2.putText(image, "AUTHORIZED PERSONNEL", (15, 140), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
    
    return image

def show_image(image):
    """
    Display the image in a fullscreen window.
    """
    setup_fullscreen_window()
    
    # Show the image
    cv2.imshow("Detection", image)
    
    # Wait for a key press
    cv2.waitKey(1)
    
    # Destroy all windows
    cv2.destroyAllWindows()

    time.sleep(0.05)


def is_aggressive(image):
    """
    Placeholder function for aggressive behavior detection.
    This function should be replaced with actual logic.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert image to RGB (required for MediaPipe)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Pose
    results = pose.process(rgb_image)


    # Extract key landmarks
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Convert normalized coordinates to Y positions (lower value means higher position in the image)
    def get_y_position(landmark):
        return landmark.y  

    lw_y, rw_y, ls_y, rs_y = map(get_y_position, [left_wrist, right_wrist, left_shoulder, right_shoulder])

    # Attack detection heuristic:
    # - Hands are above shoulders (potential attack posture)
    hands_above_shoulder = lw_y < ls_y or rw_y < rs_y


    # For now, we will just return False for both aggressive and person
    return hands_above_shoulder