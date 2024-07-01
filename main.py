import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    length_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    length_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    cos_angle = dot_product / (length_ba * length_bc)
    angle = math.acos(cos_angle)
    
    return math.degrees(angle)

def draw_angle_arc(image, point1, point2, point3):
    center = point2
    radius = 50
    angle_degrees = calculate_angle(point1, point2, point3)
    angle_degrees_text = round(angle_degrees)
    
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])
    start_angle = math.degrees(math.atan2(vector1[1], vector1[0]))
    end_angle = math.degrees(math.atan2(vector2[1], vector2[0]))
    
    if start_angle < 0:
        start_angle += 360
    if end_angle < 0:
        end_angle += 360
    
    if end_angle < start_angle:
        end_angle += 360
    
    cv2.ellipse(image, center, (radius, radius), 0, start_angle, start_angle + angle_degrees, (255, 0, 0), 2)

    text_position = (center[0] + radius, center[1] - radius)
    
    cv2.putText(image, f'{angle_degrees_text}deg', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, f'{angle_degrees_text}deg', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            custom_color = (0, 0, 255)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            custom_lines = [(23, 25), (25, 27), (24, 26), (26, 28), (11, 13), (13, 15), (12, 14), (14, 16)]
            points = {}
            for line in custom_lines:
                start_idx, end_idx = line
                start_point = results.pose_landmarks.landmark[start_idx]
                end_point = results.pose_landmarks.landmark[end_idx]
                start_coords = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
                end_coords = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
                cv2.line(image, start_coords, end_coords, custom_color, 2)
                points[start_idx] = start_coords
                points[end_idx] = end_coords

            draw_angle_arc(image, points[23], points[25], points[27])
            draw_angle_arc(image, points[24], points[26], points[28])
            draw_angle_arc(image, points[15], points[13], points[11])
            draw_angle_arc(image, points[16], points[14], points[12])

        cv2.imshow('Annotated Image', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
