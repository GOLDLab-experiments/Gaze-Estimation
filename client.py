import socket

class client:
	def __init__(self):
		self.server_addr = 'localhost'
		self.server_port = 12345

	def connect_to_server(self) -> bool:
		try:
			# Try to establish a TCP connection to the server.
			self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.sock.connect((self.server_addr, self.server_port))
			return True
		except:
			# If connection fails, print an error message and return False.
			print("Error: Connection to server failed")
			return False
		
	def is_looking_at_cameta(self) -> bool:
		# Activate the server's routine.
		self.sock.send(b'1')

		# Receive the result from the server - a boolean value.
		result = self.sock.recv(1024).decode()
		result = result.asType(bool)
		return result