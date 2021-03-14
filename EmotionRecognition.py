from facial_emotion_recognition import EmotionRecognition

import cv2, imutils

er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(1)

while True:
    success, frame = cam.read()
    frame = imutils.resize(frame, width = 600)
    frame = er.recognise_emotion(frame, return_type = 'BGR')
    cv2.imshow('Emotion Recognition', frame)
    key =cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
