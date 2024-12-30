import client

client = client.client()

connected = client.connect_to_server()
print (f"Connected to server: {connected}")

if connected:
	looking = client.is_looking_at_cameta()
	print (f"Is looking at camera: {looking}")