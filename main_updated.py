import cv2
import mediapipe as mp
import numpy as np
import math
import time

# -- CONFIGURATION -----------------------------------------------------------
# Yaw threshold in degrees.  Faces within +/-YAW_THRESHOLD of centre are
# labelled "Front Face".  Tune after live testing.
YAW_THRESHOLD_DEG = 20

# MediaPipe landmark indices used as 2-D input points for solvePnP.
#   1  -> nose tip (anchor)   152 -> chin
#  263 -> left eye outer      33  -> right eye outer
#  287 -> left mouth corner   57  -> right mouth corner
LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# 3-D reference positions on a canonical adult face (millimetres).
MODEL_POINTS_3D = np.array([
    [  0.0,    0.0,    0.0],
    [  0.0, -330.0,  -65.0],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)


# -- HELPERS -----------------------------------------------------------------

def build_camera_matrix(frame_w, frame_h):
    """Approximate camera intrinsics (focal_length = frame_width, no distortion)."""
    f  = float(frame_w)
    camera_matrix = np.array(
        [[f, 0.0, frame_w / 2.0],
         [0.0, f, frame_h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def rotation_vec_to_euler(rvec):
    """
    Convert Rodrigues rotation vector to (pitch, yaw, roll) in degrees.
    Sign convention after cv2.flip(frame, 1):
      yaw > 0 -> face turned to subject LEFT  (right in mirrored view)
      yaw < 0 -> face turned to subject RIGHT
    If live testing shows labels swapped, negate yaw in the classifier below.
    """
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2( R[2, 1],  R[2, 2]))
        yaw   = math.degrees(math.atan2(-R[2, 0],  sy))
        roll  = math.degrees(math.atan2( R[1, 0],  R[0, 0]))
    else:   # gimbal lock
        pitch = math.degrees(math.atan2(-R[1, 2],  R[1, 1]))
        yaw   = math.degrees(math.atan2(-R[2, 0],  sy))
        roll  = 0.0
    return pitch, yaw, roll


def draw_pose_axes(frame, rvec, tvec, cam_mtx, dist, origin_2d, length=80.0):
    """Draw X (red) / Y (green) / Z (blue) axes on the nose tip."""
    pts3d = np.array([[length,0,0],[0,length,0],[0,0,length]], dtype=np.float64)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, cam_mtx, dist)
    o = tuple(map(int, origin_2d))
    cv2.line(frame, o, tuple(proj[0].ravel().astype(int)), (0,   0, 255), 3)
    cv2.line(frame, o, tuple(proj[1].ravel().astype(int)), (0, 255,   0), 3)
    cv2.line(frame, o, tuple(proj[2].ravel().astype(int)), (255,  0,   0), 3)


# -- SETUP -------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(
        "Cannot open camera at index 0. "
        "Check that a webcam is connected and not in use by another application."
    )
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_matrix, dist_coeffs = build_camera_matrix(frame_w, frame_h)
prev_time = time.time()

# -- MAIN LOOP ---------------------------------------------------------------
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(rgb_frame)
        h, w, _   = frame.shape

        # Reset per-frame state so stale values never leak between frames.
        pose_ok = False
        orientation = ""
        box_color   = (200, 200, 200)
        yaw = pitch = roll = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Fix: reset pose_ok per iteration so success on face N
                # never bleeds into face N+1 if max_num_faces > 1.
                pose_ok = False

                # 2-D pixel coords for the chosen landmarks.
                image_pts = np.array(
                    [[face_landmarks.landmark[i].x * w,
                      face_landmarks.landmark[i].y * h]
                     for i in LANDMARK_IDS],
                    dtype=np.float64,
                )

                # Solve for rotation.
                ok, rvec, tvec = cv2.solvePnP(
                    MODEL_POINTS_3D, image_pts,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                if not ok:
                    # solvePnP failed -- do NOT treat as zero degrees.
                    orientation = "Pose Error"
                    box_color   = (0, 0, 255)
                else:
                    pose_ok = True
                    pitch, yaw, roll = rotation_vec_to_euler(rvec)

                    # Classify by yaw angle.
                    # To correct a left/right swap after live testing:
                    # negate yaw in both comparisons here only.
                    if abs(yaw) < YAW_THRESHOLD_DEG:
                        orientation = "Front Face"
                        box_color   = (0, 255,   0)
                    elif yaw < -YAW_THRESHOLD_DEG:
                        orientation = "Left Side Face"
                        box_color   = (0, 165, 255)
                    else:
                        orientation = "Right Side Face"
                        box_color   = (0, 165, 255)

                # Bounding box.
                x_min = int(min(lm.x for lm in face_landmarks.landmark) * w)
                x_max = int(max(lm.x for lm in face_landmarks.landmark) * w)
                y_min = int(min(lm.y for lm in face_landmarks.landmark) * h)
                y_max = int(max(lm.y for lm in face_landmarks.landmark) * h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(frame, orientation,
                            (x_min, max(y_min - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2,
                            cv2.LINE_AA)

                # Pose axes only when solve succeeded.
                if pose_ok:
                    nose_2d = (int(image_pts[0, 0]), int(image_pts[0, 1]))
                    draw_pose_axes(frame, rvec, tvec,
                                   camera_matrix, dist_coeffs, nose_2d)

                # Face mesh overlay.
                mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=1, circle_radius=1),
                )

        # HUD -- top-left corner.
        cur_time  = time.time()
        fps       = 1.0 / max(cur_time - prev_time, 1e-9)
        prev_time = cur_time

        # Show N/A when pose estimation failed so 0.0 is never mistaken
        # for a valid frontal reading.
        if pose_ok:
            yaw_str   = f"{yaw:+6.1f} deg"
            pitch_str = f"{pitch:+6.1f} deg"
            roll_str  = f"{roll:+6.1f} deg"
        else:
            yaw_str = pitch_str = roll_str = "   N/A"

        hud = [
            (f"Yaw:   {yaw_str}",  (255, 255, 255)),
            (f"Pitch: {pitch_str}", (255, 255, 255)),
            (f"Roll:  {roll_str}",  (255, 255, 255)),
            (f"FPS:   {fps:5.1f}",  (255, 255,   0)),
        ]
        for idx, (text, colour) in enumerate(hud):
            cv2.putText(frame, text, (10, 30 + idx * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

        cv2.imshow("Face Orientation Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# -- CLEANUP -----------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
