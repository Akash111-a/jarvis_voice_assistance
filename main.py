
import pyttsx3
import webbrowser
import wikipedia
import datetime
import speech_recognition as sr
import cv2
import pyautogui
# from image_gen import generate_images

import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# import nltk




engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()



def wishme():
    hour = int(datetime.datetime.now().hour)
    if hour>=6 and hour<12:
        speak("Good morning")
    elif hour>=12 and hour<16:
        speak("Good afternoon")
    elif hour>=16 and hour<18:
        speak("Good evening")
    else:
        speak("Good night")

    speak(" i am Jarvis, how can i help you ")
    
    
def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listeening....")
        r.pause_threshold = 0.5
        audio = r.listen(source)

    try:
        print("Recognizing....")
        # query = r.recognize_google(audio, language=('english-india') )
        query = r.recognize_google(audio,language='en-in')
        print(f"user said: {query}\n")


    except Exception as e:
        # print(e)
        print("say that again please.......")
        speak("say that, again please.......")
        return "none"

    return query


if __name__ == "__main__":
    wishme()
    takecommand()

    while True:
        query = takecommand().lower()

        if 'open youtube' in query:
            webbrowser.open("youtube.com")

        elif 'open chrome' in query:
            speak("initiate command")
            webbrowser.open("google.com")

        elif 'who is dushmant' in query:
            speak("dushmant is a ")

        elif 'open AI' in query:
            webbrowser.open("chatgpt")

        elif 'open camera' in query:
            speak("initiate camera comand, activate")
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xff == ord('c'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif 'hii' in query:
            speak("akash is a don")

        elif 'who is extra' in query:
            speak("extra is lauda")
        
        elif 'who am i' in query:
            # speak("")
            webbrowser.open("https://www.instagram.com/_this_is_akash___")
            speak("this is akash")

        elif 'wikipedia' in query:
            speak('searching wikipedia....')
            query = query.replace("wikipedia", "")
            Result = wikipedia.summary(query, sentences=2)
            speak("According to wikipedia")
            print(result)
            speak(Result)

        elif 'turn of' in query:
            speak("turn off comand initiate")
            pyautogui.press('ctrl+a')

        elif 'google' in query:
            query = query.replace("google", "")
            Result = webbrowser.open("chrome")

        elif 'about manas' in query:
            speak("manas is dakayat")

        elif 'search google' in query:
            query = query.replace("search google", "")
            webbrowser.open("https://www.google.com/search?q="+query)
            

        elif 'open facebook' in query:
            speak("initiate facebook comand, activate")
            webbrowser.open("facebook.com")

        elif 'open instagram' in query:
            speak("initiate  comand, activate")
            webbrowser.open("instagram.com")
            
        elif 'kutte' in query:
            speak("yes sir")

        elif 'activate doctor strange' in query:
            speak("activate doctor strange mode, initiate")
            import cv2
            import mediapipe as mpq

            mpHands=mp.solutions.hands
            hands=mpHands.Hands()
            mpDraw=mp.solutions.drawing_utils

            video=cv2.VideoCapture(0)

            video.set(3, 1400)
            video.set(4, 1000)

            img_1 = cv2.imread('magic_circles\magic_circle_ccw.png', -1)
            img_2 = cv2.imread('magic_circles\magic_circle_cw.png', -1)

            deg=0

            def position_data(lmlist):
                global wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip
                wrist = (lmlist[0][0], lmlist[0][1])
                thumb_tip = (lmlist[4][0], lmlist[4][1])
                index_mcp = (lmlist[5][0], lmlist[5][1])
                index_tip = (lmlist[8][0], lmlist[8][1])
                midle_mcp = (lmlist[9][0], lmlist[9][1])
                midle_tip = (lmlist[12][0], lmlist[12][1])
                ring_tip  = (lmlist[16][0], lmlist[16][1])
                pinky_tip = (lmlist[20][0], lmlist[20][1])

            def draw_line(p1, p2, size=5):
                cv2.line(img, p1, p2, (50,50,255), size)
                cv2.line(img, p1, p2, (255, 255, 255), round(size / 2))

            def calculate_distance(p1,p2):
                x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
                lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
                return lenght

            def transparent(targetImg, x, y, size=None):
                if size is not None:
                    targetImg = cv2.resize(targetImg, size)

                newFrame = img.copy()
                b, g, r, a = cv2.split(targetImg)
                overlay_color = cv2.merge((b, g, r))
                mask = cv2.medianBlur(a, 1)
                h, w, _ = overlay_color.shape
                roi = newFrame[y:y + h, x:x + w]

                img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
                img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
                newFrame[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

                return newFrame

            while True:
                ret,img=video.read()
                img=cv2.flip(img, 1)
                rgbimg=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result=hands.process(rgbimg)
                if result.multi_hand_landmarks:
                    for hand in result.multi_hand_landmarks:
                        lmList=[]
                        for id, lm in enumerate(hand.landmark):
                            h,w,c=img.shape
                            coorx, coory=int(lm.x*w), int(lm.y*h)
                            lmList.append([coorx, coory])

                        # mpDraw.draw_landmarks(img, hand, mqpHands.HAND_CONNECTIONS)
                        position_data(lmList)
                        palm = calculate_distance(wrist, index_mcp)
                        distance = calculate_distance(index_tip, pinky_tip)
                        ratio = distance / palm
                        print(ratio)
                        if (1.3>ratio>0.5):
                            draw_line(wrist, thumb_tip)
                            draw_line(wrist, index_tip)
                            draw_line(wrist, midle_tip)
                            draw_line(wrist, ring_tip)
                            draw_line(wrist, pinky_tip)
                            draw_line(thumb_tip, index_tip)
                            draw_line(thumb_tip, midle_tip)
                            draw_line(thumb_tip, ring_tip)
                            draw_line(thumb_tip, pinky_tip)
                        if (ratio > 1.3):
                                centerx = midle_mcp[0]
                                centery = midle_mcp[1]
                                shield_size = 3.0
                                diameter = round(palm * shield_size)
                                x1 = round(centerx - (diameter / 2))
                                y1 = round(centery - (diameter / 2))
                                h, w, c = img.shape
                                if x1 < 0:
                                    x1 = 0
                                elif x1 > w:
                                    x1 = w
                                if y1 < 0:
                                    y1 = 0
                                elif y1 > h:
                                    y1 = h
                                if x1 + diameter > w:
                                    diameter = w - x1
                                if y1 + diameter > h:
                                    diameter = h - y1
                                shield_size = diameter, diameter
                                ang_vel = 2.0
                                deg = deg + ang_vel
                                if deg > 360:
                                    deg = 0
                                hei, wid, col = img_1.shape
                                cen = (wid // 2, hei // 2)
                                M1 = cv2.getRotationMatrix2D(cen, round(deg), 1.0)
                                M2 = cv2.getRotationMatrix2D(cen, round(360 - deg), 1.0)
                                rotated1 = cv2.warpAffine(img_1, M1, (wid, hei))
                                rotated2 = cv2.warpAffine(img_2, M2, (wid, hei))
                                if (diameter != 0):
                                    img = transparent(rotated1, x1, y1, shield_size)
                                    img = transparent(rotated2, x1, y1, shield_size)
                # print(result)
                cv2.imshow("Image",img)
                k=cv2.waitKey(1)
                if k==ord('q'):
                    break

            video.release()
            cv2.destroyAllWindows()
        
        elif 'control mouse' in query:
            speak("initiate mouse command, activate")
            import cv2
            import mediapipe as mp
            import pyautogui

            # Initialize MediaPipe Hand module
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            mp_drawing = mp.solutions.drawing_utils

            # Get screen dimensions
            screen_width, screen_height = pyautogui.size()

            # Open the webcam
            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip the frame horizontally for natural interaction
                frame = cv2.flip(frame, 1)
                frame_height, frame_width, _ = frame.shape

                # Convert the frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Extracting index finger tip coordinates
                        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        x = int(index_finger_tip.x * frame_width)
                        y = int(index_finger_tip.y * frame_height)

                        # Map the hand coordinates to screen dimensions
                        screen_x = screen_width * (x / frame_width)
                        screen_y = screen_height * (y / frame_height)

                        # Move the mouse
                        pyautogui.moveTo(screen_x, screen_y)

                        # Detect click gesture (distance between thumb and index finger tip)
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        thumb_x = int(thumb_tip.x * frame_width)
                        thumb_y = int(thumb_tip.y * frame_height)
                        distance = ((thumb_x - x) ** 2 + (thumb_y - y) ** 2) ** 0.5

                        if distance < 40:  # Threshold for a "click" gesture
                            pyautogui.click()

                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the frame
                cv2.imshow("Hand Tracking Mouse Control", frame)

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # Release resources
            cap.release()
            cv2.destroyAllWindows()