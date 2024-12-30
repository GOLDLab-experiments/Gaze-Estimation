def is_looking_at_camera(gaze_vec: list) -> bool:
	"""
	This function receives a gaze vector and returns a boolean value indicating whether the user is looking at the camera.
	"""
	if gaze_vec == [1, 1, 1]:
		print("Received vector")
		return "True"
	else:
		return "False"