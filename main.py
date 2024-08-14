from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse
import cv2
import mediapipe as mp
import math
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import time
from collections import deque

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

compareIndex = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
open = [False] * len(compareIndex)
gesture = [
    [True, True, True, True, True, "Open Palm"],
    [False, False, True, True, True, "OK"],
    [False, False, False, False, True, "mini"],
    [True, False, False, False, False, "Thumb up"],
    [True, True, False, False, True, "Rock&Roll"],
    [False, True, True, False, False, "V"],
]
# 나라별 제스처 매핑
country_gestures = {
    "Brazil": "OK",
    "Turkey": "Thumb up",
    "Middle East": "Rock&Roll",
    "France": "V",
    "Australia": "mini",
    "Greece": "Rock&Roll",
    "UK": "mini",
    "New Zealand": "V",
    "Italy": "Reverse V",
    "China": "mini",
}

webcam_active = True
webcam_start_time = None
gesture_buffer = deque(maxlen=30)  # 최근 30프레임의 제스처를 저장
TARGET_GESTURE = None


def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2)) + math.sqrt(math.pow(y1 - y2, 2))


def process_frame(frame):
    global webcam_active, webcam_start_time, gesture_buffer
    h, w, c = frame.shape
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = my_hands.process(imgRGB)

    recognized_gesture = None

    if results.multi_hand_landmarks:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            for i in range(0, 5):
                open[i] = dist(
                    handLms.landmark[0].x,
                    handLms.landmark[0].y,
                    handLms.landmark[compareIndex[i][0]].x,
                    handLms.landmark[compareIndex[i][0]].y,
                ) < dist(
                    handLms.landmark[0].x,
                    handLms.landmark[0].y,
                    handLms.landmark[compareIndex[i][1]].x,
                    handLms.landmark[compareIndex[i][1]].y,
                )

            handedness = results.multi_handedness[idx].classification[0].label
            palm_visible = (
                (handLms.landmark[4].x > handLms.landmark[20].x)
                if handedness == "Left"
                else (
                    (handLms.landmark[4].x < handLms.landmark[20].x)
                    if handedness == "Right"
                    else False
                )
            )

            text_x = int(handLms.landmark[0].x * w)
            text_y = int(handLms.landmark[0].y * h)

            for g in gesture:
                if g[:5] == open:
                    gesture_name = g[5]
                    if gesture_name == "V" and not palm_visible:
                        gesture_name = "Reverse V"
                    cv2.putText(
                        frame,
                        gesture_name,
                        (text_x - 50, text_y - 250),
                        cv2.FONT_HERSHEY_PLAIN,
                        4,
                        (0, 0, 0),
                        4,
                    )
                    recognized_gesture = gesture_name
                    break

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # 웹캠 시작 후 3초 동안은 제스처 인식을 하지 않음
    if webcam_start_time is None or time.time() - webcam_start_time < 3:
        cv2.putText(
            frame, "Preparing...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        return frame

    # 인식된 제스처를 버퍼에 추가
    gesture_buffer.append(recognized_gesture)

    # 버퍼의 80% 이상이 목표 제스처와 일치하면 웹캠을 끔
    if gesture_buffer.count(TARGET_GESTURE) / len(gesture_buffer) > 0.8:
        webcam_active = False
        cv2.putText(
            frame,
            f"Target Gesture '{TARGET_GESTURE}' Recognized!",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    return frame


def get_stream_video():
    global webcam_active, webcam_start_time
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다!")
        return

    webcam_start_time = time.time()

    while webcam_active:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame)
        # cap.release()
        # cv2.destroyAllWindows()
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    yield (b"--frame\r\n" b"Content-Type: text/plain\r\n\r\n" + b"WEB_CAM_OFF\r\n")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "countries": country_gestures.keys()}
    )


@app.post("/set_gesture", response_class=HTMLResponse)
async def set_gesture(request: Request, country: str = Form(...)):
    global TARGET_GESTURE
    TARGET_GESTURE = country_gestures.get(country)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "countries": country_gestures.keys(),
            "selected_country": country,
        },
    )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        get_stream_video(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
