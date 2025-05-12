# Runs on omri's computer so it needs to be able to be imported
import typing
from socket import socket, AF_INET, SOCK_DGRAM

def detect_face_and_helmet() -> typing.Tuple[bool, bool]:
    s = socket(AF_INET, SOCK_DGRAM)
    ip = '127.0.0.1'  # TODO: Change this to the IP address of the C++ server on the BOON network
    port = 12345

    # send a message to the gaze and helmet server
    s.sendto(b'1', (ip, port))

    data, _ = s.recvfrom(1)
    looking_at_camera = data.decode('utf-8')

    data, _ = s.recvfrom(1)
    helmet = data.decode('utf-8')

    s.close()

    return looking_at_camera, helmet