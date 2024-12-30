# install opencv-python
import cv2

# Make sure the folders exist

def capture_video():
    stream = cv2.VideoCapture(1)

    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not stream.isOpened():
        print("No stream :(")
        exit()

    fps = stream.get(cv2.CAP_PROP_FPS)
    width = int(stream.get(3))
    height = int(stream.get(4))

    output = cv2.VideoWriter("videos/1080p.mp4",
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                fps=fps, frameSize=(width, height))

    while True:
        ret, frame = stream.read()
        if not ret:
            print("No more stream :(")
            break
        
        frame = cv2.resize(frame, (width, height))
        output.write(frame)
        cv2.imshow("Webcam!", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
    output.release()
    cv2.destroyAllWindows()

def snap_photo():
    stream = cv2.VideoCapture(1)

    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not stream.isOpened():
        print("No stream :(")
        exit()

    ret, frame = stream.read()
    if ret:
        cv2.imwrite("snapshot.jpg", frame)
        cv2.imshow("Snapshot", frame)
        # cv2.waitKey(0)

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    snap_photo()
