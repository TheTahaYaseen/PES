import cv2
import mediapipe

capture = cv2.VideoCapture(0)

capture.set(3, 640)
capture.set(4, 480)
capture.set(10, 200)

mediapipe_pose = mediapipe.solutions.pose
pose = mediapipe_pose.Pose()

mediapipe_draw = mediapipe.solutions.drawing_utils

while True:
    success, succeeding_image = capture.read()
    succeeding_image_rgb = cv2.cvtColor(succeeding_image, cv2.COLOR_BGR2RGB)

    results = pose.process(succeeding_image_rgb)
    if results.pose_landmarks:
        mediapipe_draw.draw_landmarks(succeeding_image, results.pose_landmarks, mediapipe_pose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, channel = succeeding_image.shape
            channel_x, channel_y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(succeeding_image, (channel_x, channel_y), 3, (255, 0, 0), cv2.FILLED)

    cv2.imshow("MyVid", succeeding_image)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

capture.release()

cv2.destroyAllWindows()