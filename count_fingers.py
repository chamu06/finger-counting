import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands # to detect 21 red dots/landmarks
mp_drawing = mp.solutions.drawing_utils # grey line connecting them

hands = mp_hands.Hands(min_detection_confidence = 0.8 , 
                       min_tracking_confidence = 0.5)
cap = cv2.VideoCapture(0)


tipIds = [4, 8, 12, 16, 20]
# Define a function to count fingers
def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks: # checking if any value in hand_landmarks(0 to 20 = 21 red points)
        landmarks=hand_landmarks[handNo].landmark
        #print(landmarks)
        fingers=[] # [ 1 , 0 , 1 , 1]
        for i in tipIds:
            fty=landmarks[i].y # [4, 8, 12, 16, 20]
            fby=landmarks[i-2].y # [2, 6, 10, 14, 18]
            if i !=4 : # don't do it for 4th id
                if fty<fby:
                    print("Finger is OPEN")
                    fingers.append(1)
                if fty>fby:
                    print("Finger is close")
                    fingers.append(0)
        totalFingers = fingers.count(1)
        txt = f"Total Fingers : {totalFingers}"
        # 1. where message? 
        # 2. what message ?
        # 3. position of the text - (x,y)
        # 4. style of the text 
        # 5. size of text = 1 (original 12 fontSize)
        # 6. color of the text = ( blue , green , red) = from 0 to 255
        # 6. stroke of the text
        cv2.putText(image , txt, (50,50), cv2.FONT_HERSHEY_COMPLEX,1,
                     (255,0,0),2)
    

def drawLandmarks(image , hand_landmarks):
    if hand_landmarks:
        for i in hand_landmarks:
            mp_drawing.draw_landmarks(image , i, mp_hands.HAND_CONNECTIONS)

while True:
    success, image = cap.read() # reading frame from webcam
    image = cv2.flip(image , 1) # flipping pic in opposite
    results=hands.process(image) # algo is applied on pic

    hand_landmarks = results.multi_hand_landmarks
    drawLandmarks(image , hand_landmarks)
    countFingers(image,hand_landmarks)
    cv2.imshow("Media Controller", image)
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()

