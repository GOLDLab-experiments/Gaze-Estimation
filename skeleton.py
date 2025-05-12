import cv2
import mediapipe as mp
import numpy as np

# Distance calibration (adjust these values for your camera setup)
KNOWN_DISTANCE = 1.0  # Reference distance in meters
KNOWN_WIDTH = 180  # Approximate skeleton width in pixels at 1 meter
FOCAL_LENGTH = 600  # Estimated focal length in pixels
MIN_DISTANCE = 500 # Minimum distance in pixels for human obstacle detection

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def is_aggressive(path):
    """
    Detects aggressive behavior in a given image based on body posture.
    Returns (bool, image_with_skeleton).
    """

    # Load the image
    image = cv2.imread(path)

    # Convert image to RGB (required for MediaPipe)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Pose
    results = pose.process(rgb_image)
    
    if not results.pose_landmarks:
        return '0', '0' # No person detected, return original image

    # Distance estimation based on skeleton width
    frame_height, frame_width, _ = image.shape

    keypoints = {i: results.pose_landmarks.landmark[i] for i in range(len(mp_pose.PoseLandmark))}
    
    # Compute skeleton width
    x_values = [kp.x * frame_width for kp in keypoints.values()]
    person_width = max(x_values) - min(x_values)  # Width of the detected person in pixels

    # Estimate distance
    if person_width > 0:
        estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / person_width
        print (f'The estimated distance is: {estimated_distance}')
    else:
        estimated_distance = '0' 

    # Return True if the person is too close
    #return estimated_distance is not None and estimated_distance < 150


    # Create a copy of the image to draw on
    output_image = image.copy()

    # Draw the skeleton in red
    mp_drawing.draw_landmarks(
        output_image, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),  # Red color
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
    )

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

    
    cv2.imwrite("camera_image.jpg", output_image)

    if estimated_distance is not '0' and estimated_distance < MIN_DISTANCE:
        estimated_distance = '1'
    else:
        estimated_distance = '0'

    if hands_above_shoulder:
        return '1', estimated_distance 
    else:
        return '0', estimated_distance # Return aggression status and modified image

def get_person_boundaries(image_path):
    """
    Uses skeleton detection to identify person boundaries.
    Returns (x, y, width, height) of the bounding box around the person.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            person_coords = get_person_boundaries(image_path)
            return None
        
        # First try using MediaPipe Pose which is already in use in is_aggressive
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        
        # Debug MediaPipe detection
        if not results.pose_landmarks:
            print("MediaPipe did not detect any pose landmarks")
        else:
            print(f"MediaPipe detected {len(results.pose_landmarks.landmark)} landmarks")
        
        
        if results.pose_landmarks:
            # Get frame dimensions
            height, width, _ = image.shape
            
            # Extract all landmark coordinates and convert to pixel values
            landmarks = []
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                landmarks.append((x_px, y_px))


                # Debug specific landmarks
                if i in [0, 11, 12, 23, 24]:  # Key points (nose, shoulders, hips)
                    print(f"Landmark {i}: ({x_px}, {y_px})")
            
            if not landmarks:
                print("No valid landmarks found")
                return None
                
            
            # Find the bounding box that contains all landmarks
            x_values = [x for x, y in landmarks]
            y_values = [y for x, y in landmarks]
            
            # Calculate bounding box coordinates
            min_x = max(0, min(x_values))
            min_y = max(0, min(y_values))
            max_x = min(width, max(x_values))
            max_y = min(height, max(y_values))
            
            # Calculate width and height
            w = max_x - min_x
            h = max_y - min_y
            
            # Expand the box to include more context (e.g., for helmet detection)
            # Add more space above to capture helmet area
            expansion_x = int(w * 0.15)  # 15% expansion horizontally
            expansion_y_top = int(h * 0.5)  # 50% expansion at top for helmet
            expansion_y_bottom = int(h * 0.2)  # 20% expansion at bottom
            
            x = max(0, min_x - expansion_x)
            y = max(0, min_y - expansion_y_top)
            w = min(width - x, w + 2 * expansion_x)
            h = min(height - y, h + expansion_y_top + expansion_y_bottom)
            
            print("MediaPipe detection successful")
            return (x, y, w, h)
        
        # Fallback to HOG detector if MediaPipe doesn't detect a person
        print("MediaPipe didn't detect a person, falling back to HOG detector")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Try different parameters for better detection
        # First try with default parameters
        boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        if len(boxes) == 0:
            # Try again with more lenient parameters
            boxes, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.1)
        
        if len(boxes) == 0:
            print("HOG detector failed to find a person")
            
            # Last resort: assume person is in the center of the frame
            height, width, _ = image.shape
            center_x = width // 2
            center_y = height // 2
            
            # Create a default region of interest
            roi_width = width // 2  # Use half the image width
            roi_height = height // 2  # Use half the image height
            
            x = center_x - (roi_width // 2)
            y = center_y - (roi_height // 2)
            
            print("Using default region in the center of the image")
            return (x, y, roi_width, roi_height)
        
        # Get the box with the highest confidence
        best_box = boxes[np.argmax(weights)]
        x, y, w, h = best_box
        
        # Expand the box slightly to include arms and any potential helmet
        expansion_x = int(w * 0.15)
        expansion_y_top = int(h * 0.3)
        expansion_y_bottom = int(h * 0.1)
        
        x = max(0, x - expansion_x)
        y = max(0, y - expansion_y_top)
        w = min(image.shape[1] - x, w + 2 * expansion_x)
        h = min(image.shape[0] - y, h + expansion_y_top + expansion_y_bottom)
        
        print("HOG detection successful")
        return (x, y, w, h)
        
    except Exception as e:
        print(f"Error in get_person_boundaries: {e}")
        # Fallback: use center of the image
        try:
            height, width, _ = image.shape
            center_x = width // 2
            center_y = height // 2
            roi_width = width // 2
            roi_height = height // 2
            x = center_x - (roi_width // 2)
            y = center_y - (roi_height // 2)
            print(f"Exception occurred, using default region: {e}")
            return (x, y, roi_width, roi_height)
        except:
            return None