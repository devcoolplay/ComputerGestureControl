import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui
import autopy

###########################################
wCam = 640
hCam = 480
clickPossible = True
frameR = 150 # Frame reduction # standard = 150
smoothening = 4.5
wScreen, hScreen = autopy.screen.size()
###########################################

imgGlobal= 0

plocX, plocY = 0, 0
clocX, clocY = 0, 0

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def getPosePositions(results, img):
    lmList = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
    return lmList

def getHandPositions(results, img):
    lmList = []
    if results.left_hand_landmarks:
        for id, lm in enumerate(results.left_hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
    return lmList

def fingersUp(trackingpoints):
    tipIds = [4, 8, 12, 16, 20]
    if len(trackingpoints) != 0:
        fingers = []

        # Thumb
        if trackingpoints[tipIds[0]][1] < trackingpoints[tipIds[0] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers
        for id in range(1, 5):
            if trackingpoints[tipIds[id]][2] < trackingpoints[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def moveMouse():
    pass

def checkHandFunction(trackingPoints):
    global imgGlobal, clickPossible, plocX, plocY, clocX, clocY
    middleFingerX, middleFingerY = trackingPoints[7][1:]
    up = fingersUp(trackingPoints)
    print(fingersUp(trackingPoints))
    # Move coursur
    if up[0] == 1 and up[1] == 1 and up[2] == 1 and up[3] == 1 and up[4] == 1:
        if not clickPossible:
            clickPossible = True
        cv2.rectangle(imgGlobal, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        xScreen = np.interp(middleFingerX, (frameR, wCam - frameR), (0, wScreen))
        yScreen = np.interp(middleFingerY, (frameR, hCam - frameR), (0, hScreen))
        
        # Smoothen values
        clocX = plocX + (xScreen - plocX) / smoothening
        clocY = plocY + (yScreen - plocY) / smoothening

        autopy.mouse.move(clocX, clocY)
        plocX, plocY = clocX, clocY
    # Normal left click
    elif up[0] == 0 and up[1] == 1 and up[2] == 1 and up[3] == 1 and up[4] == 1:
        if clickPossible:
            autopy.mouse.click()
            clickPossible = False
    # Double left click
    elif up[0] == 1 and up[1] == 1 and up[2] == 0 and up[3] == 1 and up[4] == 1:
        if clickPossible:
            autopy.mouse.click()
            clickPossible = False
            autopy.mouse.click()
    # Right click
    elif up[0] == 1 and up[1] == 1 and up[2] == 1 and up[3] == 1 and up[4] == 0:
        if clickPossible:
            print("left click")
            pyautogui.click(button='right')
            clickPossible = True

def isHandAboveShoulder(poseTrackingPoints, handTrackingPoints):
    if poseTrackingPoints[19][2] < poseTrackingPoints[11][2] + 50: #((poseTrackingPoints[19][2] < poseTrackingPoints[11][2]) and (poseTrackingPoints[19][1] > poseTrackingPoints[11][1])) or ((poseTrackingPoints[20][2] < poseTrackingPoints[12][2]) and (poseTrackingPoints[20][1] < poseTrackingPoints[12][1]))
        checkHandFunction(handTrackingPoints)
        #moveMouse()
        #print("activated")
    

cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        success, image = cap.read()
       
        start = time.time()


        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and detect the holistic
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True

        imgGlobal = image
        # Process tracking  points
        try:
            isHandAboveShoulder(getPosePositions(results, image), getHandPositions(results, image))
        except Exception as e:
            pass
            #print(e)

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)




        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('MediaPipe Holistic', image)


        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()