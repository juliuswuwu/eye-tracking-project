import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[0]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # hor_line = cv2.line(img, left_point, right_point, (0, 255, 0), 1)
    # ver_line = cv2.line(img, center_top, center_bottom, (0, 255, 0), 1)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio


font = cv2.FONT_HERSHEY_PLAIN
while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        # x1,y1 = face.left(), face.top()
        # x2,y2 = face.right(), face.bottom()
        # cv2.rectangle(img,(x1,y1), (x2,y2), (255,0,0),2)

        landmarks = predictor(gray, face)

        # detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blinking_ratio > 5.7:
            cv2.putText(img, "both Blinking", (50, 150), font, 3, (255, 0, 0))
        elif left_eye_ratio > 5.7:
            cv2.putText(img, "left Blinking", (50, 150), font, 3, (255, 0, 0))
        elif right_eye_ratio > 5.7:
            cv2.putText(img, "right Blinking", (50, 150), font, 3, (255, 0, 0))

        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)
                                    ], np.int32)

        # cv2.polylines(img, [left_eye_region], True, (0,0,255), 1)

        height, width, _ = img.shape
        mask = np.zeros((height, width), np.uint8)

        cv2.polylines(mask, [left_eye_region], True, 255, 1)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        eye = cv2.resize(gray_eye, None, fx=5, fy=5)
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)

        cv2.imshow("Eye", eye)
        cv2.imshow("threshold", threshold_eye)
        cv2.imshow("left eye", left_eye)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
