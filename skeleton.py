import cv2
import mediapipe as mp
import numpy as np

class SkeletonDetector:
    def __init__(self):
        """
        Initializes the SkeletonDetector with MediaPipe Pose and other configurations.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()

    def detect_landmarks(self, image):
        """
        Detects landmarks in the given image and draws the skeleton.
        Returns the landmarks and the image with the skeleton drawn.
        """
        # Convert image to RGB (required for MediaPipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Pose
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None  # No person detected
        
        # Create a copy of the image to draw on
        output_image = image.copy()

        # Draw the skeleton in red
        self.mp_drawing.draw_landmarks(
            output_image, 
            results.pose_landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),  # Red color
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
        )

        return results.pose_landmarks

    def get_person_boundaries(self, image):
        """
        Uses skeleton detection to identify person boundaries.
        Returns (x, y, width, height) of the bounding box around the person.
        """
        try:
            if image is None:
                return None
            
            # Detect landmarks using MediaPipe Pose
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                # Calculate bounding box directly
                height, width, _ = image.shape
                landmarks = [
                    (int(landmark.x * width), int(landmark.y * height))
                    for landmark in results.pose_landmarks.landmark
                ]

                x_values = [x for x, y in landmarks]
                y_values = [y for x, y in landmarks]

                min_x = max(0, min(x_values))
                min_y = max(0, min(y_values))
                max_x = min(width, max(x_values))
                max_y = min(height, max(y_values))

                w = max_x - min_x
                h = max_y - min_y

                # Expand the box
                expansion_x = int(w * 0.15)
                expansion_y_top = int(h * 0.5)
                expansion_y_bottom = int(h * 0.2)

                x = max(0, min_x - expansion_x)
                y = max(0, min_y - expansion_y_top)
                w = min(width - x, w + 2 * expansion_x)
                h = min(height - y, h + expansion_y_top + expansion_y_bottom)

                return (x, y, w, h)
            
            # Fallback to HOG detector directly
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(4, 4), scale=1.05)

            if len(boxes) == 0:
                boxes, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.1)
            
            if len(boxes) == 0:
                # Fallback to center of the image
                height, width, _ = image.shape
                center_x = width // 2
                center_y = height // 2
                roi_width = width // 2
                roi_height = height // 2

                x = center_x - (roi_width // 2)
                y = center_y - (roi_height // 2)

                return (x, y, roi_width, roi_height)

            best_box = boxes[np.argmax(weights)]
            x, y, w, h = best_box

            # Expand the box
            expansion_x = int(w * 0.15)
            expansion_y_top = int(h * 0.3)
            expansion_y_bottom = int(h * 0.1)

            x = max(0, x - expansion_x)
            y = max(0, y - expansion_y_top)
            w = min(image.shape[1] - x, w + 2 * expansion_x)
            h = min(image.shape[0] - y, h + expansion_y_top + expansion_y_bottom)

            return (x, y, w, h)
        
        except Exception as e:
            print(f"Error in get_person_boundaries: {e}")
            # Fallback to center of the image
            height, width, _ = image.shape
            center_x = width // 2
            center_y = height // 2
            roi_width = width // 2
            roi_height = height // 2

            x = center_x - (roi_width // 2)
            y = center_y - (roi_height // 2)

            return (x, y, roi_width, roi_height)
        
"""
# Example usage
detector = SkeletonDetector()

# Load an image
image = cv2.imread("path/to/image.jpg")

# Detect landmarks and draw skeleton
landmarks, skeleton_image = detector.detect_landmarks(image)

# Get person boundaries
boundaries = detector.get_person_boundaries(image)
print("Person boundaries:", boundaries)
"""