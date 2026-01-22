# ===================== IMPORTS =====================
import pyttsx3
import webbrowser
import wikipedia
import datetime
import speech_recognition as sr
import nltk
import pyautogui 
import pywhatkit
import psutil
import time
import threading
import re
import time

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== NLTK AUTO SETUP =====================
def ensure_nltk_data():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

ensure_nltk_data()

# ===================== VOICE ENGINE =====================
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ===================== WISH =====================
def wishme():
    hour = datetime.datetime.now().hour
    if 6 <= hour < 12:
        speak("Good morning")
    elif 12 <= hour < 16:
        speak("Good afternoon")
    elif 16 <= hour < 19:
        speak("Good evening")
    else:
        speak("Good night")
    speak("I am Jarvis. How can I help you?")

# ===================== SPEECH INPUT =====================
def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio, language='en-in')
        print("You said:", query)
        return query.lower()
    except:
        return ""

# ===================== CHATBOT =====================
def load_dataset(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                q, a = line.split(':', 1)
                data.append({'question': q, 'answer': a})
    return data

def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(tokens)

def chatbot_reply(text):
    dataset = load_dataset(r"C:\Users\jenaa\Desktop\p\data.txt")
    corpus = [preprocess_text(i['question']) for i in dataset]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    q_vec = vectorizer.transform([preprocess_text(text)])
    index = cosine_similarity(q_vec, X).argmax()
    return dataset[index]['answer']

# ===================== BATTERY =====================
def check_battery():
    percent = int(psutil.sensors_battery().percent)
    speak(f"The device is running on {percent} percent power")

# ===================== VOLUME BY PERCENTAGE =====================
def set_volume_percentage(target):
    target = max(0, min(100, target))
    pyautogui.press('volumemute')
    time.sleep(0.1)
    for _ in range(target // 2):
        pyautogui.press('volumeup')

def handle_volume_percentage(text):
    match = re.search(r'(\d+)\s*percent', text)
    if match:
        percent = int(match.group(1))
        speak(f"Setting volume to {percent} percent")
        set_volume_percentage(percent)
        return True
    return False

# ===================== ADVANCED MEDIA CONTROLS =====================
def volume_up(): pyautogui.press('up')
def volume_down(): pyautogui.press('down')
def seek_forward(): pyautogui.press('right')
def seek_backward(): pyautogui.press('left')
def seek_forward_10s(): pyautogui.press('l')
def seek_backward_10s(): pyautogui.press('j')
def seek_backward_frame(): pyautogui.press(',')
def seek_forward_frame(): pyautogui.press('.')
def seek_to_beginning(): pyautogui.press('home')
def seek_to_end(): pyautogui.press('end')
def seek_to_previous_chapter(): pyautogui.hotkey('ctrl', 'left')
def seek_to_next_chapter(): pyautogui.hotkey('ctrl', 'right')
def decrease_playback_speed(): pyautogui.hotkey('shift', ',')
def increase_playback_speed(): pyautogui.hotkey('shift', '.')
def move_to_next_video(): pyautogui.hotkey('shift', 'n')
def move_to_previous_video(): pyautogui.hotkey('shift', 'p')

def perform_media_action(text):
    if handle_volume_percentage(text):
        return

    if "volume up" in text or "volume badhao" in text:
        volume_up()
    elif "volume down" in text or "volume ghatao" in text:
        volume_down()
    elif "seek forward 10 seconds" in text:
        seek_forward_10s()
    elif "seek backward 10 seconds" in text:
        seek_backward_10s()
    elif "seek forward frame" in text:
        seek_forward_frame()
    elif "seek backward frame" in text:
        seek_backward_frame()
    elif "seek forward" in text:
        seek_forward()
    elif "seek backward" in text:
        seek_backward()
    elif "seek to beginning" in text:
        seek_to_beginning()
    elif "seek to end" in text:
        seek_to_end()
    elif "previous chapter" in text:
        seek_to_previous_chapter()
    elif "next chapter" in text:
        seek_to_next_chapter()
    elif "increase playback speed" in text:
        increase_playback_speed()
    elif "decrease playback speed" in text:
        decrease_playback_speed()
    elif "next video" in text:
        move_to_next_video()
    elif "previous video" in text:
        move_to_previous_video()
    elif "play" in text or "pause" in text or "stop" in text:
        pyautogui.press('space')

# ===================== AUTO-SKIP YOUTUBE ADS =====================
def auto_skip_ads():
    while True:
        time.sleep(6)
        pyautogui.press('k')
        time.sleep(0.3)
        pyautogui.press('tab')
        pyautogui.press('tab')
        pyautogui.press('enter')

def start_auto_skip():
    threading.Thread(target=auto_skip_ads, daemon=True).start()

# ===================== YOUTUBE SEARCH =====================
def interactive_youtube_search():
    speak("What should I search on YouTube?")
    query = takecommand()
    if query:
        speak(f"Searching YouTube for {query}")
        pywhatkit.playonyt(query)
        time.sleep(5)
        start_auto_skip()

def play_first_video():
    speak("Playing first video")
    pyautogui.press('1')

def play_second_video():
    speak("Playing second video")
    pyautogui.press('2')

# ===================== MAIN =====================
if __name__ == "__main__":
    wishme()

    while True:
        query = takecommand()
        if not query:
            continue

        if "exit" in query or "quit" in query:
            speak("Goodbye")
            break

        elif "search on youtube" in query:
            interactive_youtube_search()

        elif "play first video" in query or "play 1st video" in query:
            play_first_video()

        elif "play second video" in query or "play 2nd video" in query:
            play_second_video()

        elif "battery" in query:
            check_battery()

        elif query.startswith("search"):
            pyautogui.hotkey("/")
            query = query.replace("search", "")
            pyautogui.write(query)
            time.sleep(3)
            pyautogui.press("enter")
            speak(f"searching {query}")
            print(f"searching {query}")

        # elif "minimise" in query or "minimise the window " in query or " hata do " in query:
        #     speak("minimising window")
        #     pyautogui.hotkey('win', 'down')
        #     # gui.hotkey(*args: 'win', 'down')

        # elif "maximize" in query or "maximize the window " in query or " bada karo " in query:
        #     speak("maximize window")
        #     pyautogui.hotkey('win', 'up')

        elif "minimise" in query or "minimize" in query or "hata do" in query:
            speak("Minimizing window")
            pyautogui.hotkey('win', 'down')
        
        elif "maximize" in query or "maximize the window" in query or "bada karo" in query:
            speak("Maximizing window")
            pyautogui.hotkey('win', 'down')
            time.sleep(0.2)
            pyautogui.hotkey('win', 'up')
        
        elif "restore" in query or "normal size" in query:
            speak("Restoring window")
            pyautogui.hotkey('win', 'down')


        elif query.startswith("open"):
            site = query.replace("open", "").strip()
            speak(f"Opening {site}")
            webbrowser.open(f"https://{site}.com")

        elif "wikipedia" in query:
            try:
                speak(wikipedia.summary(query.replace("wikipedia", ""), sentences=2))
            except:
                speak("No result found")

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

        else:
            perform_media_action(query)
            speak(chatbot_reply(query))