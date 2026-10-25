import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert the image to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract x-coordinates of key landmarks
                # Left eye: 33, 133; Right eye: 362, 263; Nose: 1
                left_eye_x = face_landmarks.landmark[33].x
                right_eye_x = face_landmarks.landmark[263].x
                nose_x = face_landmarks.landmark[1].x

                # Calculate distance differences
                left_to_nose = abs(left_eye_x - nose_x)
                right_to_nose = abs(right_eye_x - nose_x)

                # Draw bounding box around the face
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Determine face orientation
                if abs(left_to_nose - right_to_nose) < 0.05:  # Threshold for symmetry
                    orientation = "Front Face"
                else:
                    orientation = "Side Face"

                # Display the orientation above the face
                cv2.putText(frame, orientation, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Optionally draw the face landmarks
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                )

        # Display the frame
        cv2.imshow('Face Orientation Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
