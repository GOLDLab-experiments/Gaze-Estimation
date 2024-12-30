import camera
import vector_algorithm
import calculation

import socket

server_addr = 'localhost'
server_port = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((server_addr, server_port))
s.listen(1)
print(f"Server is listening on adress {server_addr} and port {server_port}.")
client_socket, client_info = s.accept()

while True: # Maybe change so that when the main program stops, the server stops too
    signal = client_socket.read(1).decode()
    if signal:
        # For debugging, print a messege
        print("Received signal")
        
		# Run the routine
        image, faces_matrix = camera.snap_photo()
        gaze_vec = vector_algorithm.estimate_gaze(image, faces_matrix)
        looking = calculation.is_looking_at_camera(gaze_vec)
        
		# Send the result back to the client
        client_socket.send(looking.encode())

print("Server stopped.")
s.close()