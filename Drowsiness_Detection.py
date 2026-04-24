from pathlib import Path
import os
import time

import cv2
import numpy as np
from scipy.spatial import distance


THRESH = float(os.getenv("DROWSY_EAR_THRESH", "0.28"))
SLEEP_SECONDS = float(os.getenv("DROWSY_SLEEP_SECONDS", "3"))
ALERT_REPEAT_SECONDS = float(os.getenv("DROWSY_ALERT_REPEAT_SECONDS", "1.0"))
MODEL_PATH = Path(__file__).resolve().parent / "models" / "shape_predictor_68_face_landmarks.dat"
HAAR_FACE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
HAAR_EYE = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"


def eye_aspect_ratio(eye):
	a = distance.euclidean(eye[1], eye[5])
	b = distance.euclidean(eye[2], eye[4])
	c = distance.euclidean(eye[0], eye[3])
	if c == 0:
		return 0.0
	return (a + b) / (2.0 * c)


def draw_alert(frame):
	cv2.putText(
		frame,
		"****************ALERT!****************",
		(10, 30),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(0, 0, 255),
		2,
	)


def play_alert_sound():
	# Keep this dependency-free; on Windows this gives an audible notification.
	try:
		import winsound

		winsound.Beep(2000, 400)
	except Exception:
		try:
			import winsound

			winsound.MessageBeep(winsound.MB_ICONHAND)
		except Exception:
			# Terminal bell fallback for non-Windows or restricted environments.
			print("\a", end="")


def draw_debug(frame, status_text):
	cv2.putText(
		frame,
		status_text,
		(10, frame.shape[0] - 15),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(50, 220, 255),
		1,
	)


def open_webcam():
	# Try a few backends because Windows camera APIs vary by driver/policy.
	for backend in (None, cv2.CAP_DSHOW, cv2.CAP_MSMF):
		cap = cv2.VideoCapture(0) if backend is None else cv2.VideoCapture(0, backend)
		if cap.isOpened():
			return cap
		cap.release()
	raise RuntimeError("Could not open webcam")


def run_with_dlib():
	import dlib
	from imutils import face_utils

	if not MODEL_PATH.exists():
		raise FileNotFoundError("Missing landmark model at models/shape_predictor_68_face_landmarks.dat")

	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor(str(MODEL_PATH))
	(l_start, l_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(r_start, r_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

	cap = open_webcam()

	closed_since = None
	last_alarm_time = 0.0
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame = cv2.resize(frame, (450, int(frame.shape[0] * (450.0 / frame.shape[1]))))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = detect(gray, 0)
		ear = None

		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)
			left_eye = shape[l_start:l_end]
			right_eye = shape[r_start:r_end]
			left_ear = eye_aspect_ratio(left_eye)
			right_ear = eye_aspect_ratio(right_eye)
			ear = (left_ear + right_ear) / 2.0

			left_eye_hull = cv2.convexHull(left_eye)
			right_eye_hull = cv2.convexHull(right_eye)
			cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
			break

		now = time.perf_counter()
		if ear is not None and ear < THRESH:
			if closed_since is None:
				closed_since = now
			closed_duration = now - closed_since
			if closed_duration >= SLEEP_SECONDS:
				draw_alert(frame)
				if now - last_alarm_time >= ALERT_REPEAT_SECONDS:
					play_alert_sound()
					last_alarm_time = now
		else:
			closed_since = None
			closed_duration = 0.0

		if ear is None:
			draw_debug(frame, f"Face not detected  Closed: 0.0/{SLEEP_SECONDS:.1f}s")
		else:
			draw_debug(frame, f"EAR: {ear:.3f}  THRESH: {THRESH:.2f}  Closed: {closed_duration:.1f}/{SLEEP_SECONDS:.1f}s")

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


def run_with_mediapipe():
	import mediapipe as mp

	mp_face_mesh = mp.solutions.face_mesh
	left_eye_idxs = np.array([362, 385, 387, 263, 373, 380], dtype=np.int32)
	right_eye_idxs = np.array([33, 160, 158, 133, 153, 144], dtype=np.int32)

	cap = open_webcam()

	closed_since = None
	last_alarm_time = 0.0
	with mp_face_mesh.FaceMesh(
		static_image_mode=False,
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
	) as face_mesh:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			frame = cv2.resize(frame, (450, int(frame.shape[0] * (450.0 / frame.shape[1]))))
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = face_mesh.process(rgb)
			now = time.perf_counter()

			ear = None
			if results.multi_face_landmarks:
				landmarks = results.multi_face_landmarks[0].landmark
				pts = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in landmarks], dtype=np.float32)

				left_eye = pts[left_eye_idxs]
				right_eye = pts[right_eye_idxs]
				left_ear = eye_aspect_ratio(left_eye)
				right_ear = eye_aspect_ratio(right_eye)
				ear = (left_ear + right_ear) / 2.0

				left_eye_hull = cv2.convexHull(left_eye.astype(np.int32))
				right_eye_hull = cv2.convexHull(right_eye.astype(np.int32))
				cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

			if ear is not None and ear < THRESH:
				if closed_since is None:
					closed_since = now
				closed_duration = now - closed_since
				if closed_duration >= SLEEP_SECONDS:
					draw_alert(frame)
					if now - last_alarm_time >= ALERT_REPEAT_SECONDS:
						play_alert_sound()
						last_alarm_time = now
			else:
				closed_since = None
				closed_duration = 0.0

			if ear is None:
				draw_debug(frame, f"Face not detected  Closed: 0.0/{SLEEP_SECONDS:.1f}s")
			else:
				draw_debug(frame, f"EAR: {ear:.3f}  THRESH: {THRESH:.2f}  Closed: {closed_duration:.1f}/{SLEEP_SECONDS:.1f}s")

			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

	cap.release()
	cv2.destroyAllWindows()


def run_with_haar():
	if not HAAR_FACE.exists() or not HAAR_EYE.exists():
		raise FileNotFoundError("OpenCV Haar cascade files are missing")

	face_cascade = cv2.CascadeClassifier(str(HAAR_FACE))
	eye_cascade = cv2.CascadeClassifier(str(HAAR_EYE))
	if face_cascade.empty() or eye_cascade.empty():
		raise RuntimeError("Failed to load OpenCV Haar cascade classifiers")

	cap = open_webcam()

	closed_since = None
	last_alarm_time = 0.0
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame = cv2.resize(frame, (450, int(frame.shape[0] * (450.0 / frame.shape[1]))))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

		eyes_detected = 0
		for (x, y, w, h) in faces:
			face_roi_gray = gray[y : y + h, x : x + w]
			face_roi_color = frame[y : y + h, x : x + w]
			eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=6)

			for (ex, ey, ew, eh) in eyes[:2]:
				eyes_detected += 1
				cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 150, 0), 1)
			break

		now = time.perf_counter()
		# Approximate drowsiness on Haar: if eyes are missing continuously, treat as closed-eye duration.
		if eyes_detected < 2:
			if closed_since is None:
				closed_since = now
			closed_duration = now - closed_since
			if closed_duration >= SLEEP_SECONDS:
				draw_alert(frame)
				if now - last_alarm_time >= ALERT_REPEAT_SECONDS:
					play_alert_sound()
					last_alarm_time = now
		else:
			closed_since = None
			closed_duration = 0.0

		draw_debug(frame, f"Eyes: {eyes_detected}  Closed: {closed_duration:.1f}/{SLEEP_SECONDS:.1f}s")

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


def main():
	try:
		run_with_dlib()
	except Exception as exc:
		print(f"[INFO] dlib backend unavailable ({exc}). Trying MediaPipe backend.")
		try:
			run_with_mediapipe()
		except Exception as exc2:
			print(f"[INFO] MediaPipe backend unavailable ({exc2}). Falling back to Haar-cascade backend.")
			run_with_haar()


if __name__ == "__main__":
	main()
