import cv2

# From detection client
def setup_fullscreen_window():
    """
    Create a named window in fullscreen mode
    """
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
