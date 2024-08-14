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

# Initialize FastAPI application
app = FastAPI()

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize MediaPipe Hand Tracking model and utilities
mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 손가락 특정 랜드마크 인덱스
compareIndex = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

# 다섯손가락의 펴짐 상태
open = [False] * len(compareIndex)

# 나라별 제스처 매핑(손가락 펴짐 유무 포함)
gesture = [
    [True, True, True, True, True, "Open Palm"],
    [False, False, True, True, True, "OK"],
    [False, False, False, False, True, "mini"],
    [True, False, False, False, False, "Thumb up"],
    [True, True, False, False, True, "Rock&Roll"],
    [False, True, True, False, False, "V"],
]
# 나라별 제스처 매핑(Target Gesture 설정 위해)
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

# 웹캠 상태 변수
webcam_active = True
webcam_start_time = None
# 제스처 인식 버퍼 지정
gesture_buffer = deque(maxlen=30)  # 최근 30프레임의 제스처를 저장
# 목표 제스처 변수 초기 설정
TARGET_GESTURE = None


# 거리 계산 함수
def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2)) + math.sqrt(math.pow(y1 - y2, 2))


# 프레임 처리 함수
def process_frame(frame):
    global webcam_active, webcam_start_time, gesture_buffer
    h, w, c = frame.shape
    # RGR -> RGB 색상 변경
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 손 랜드마크 인식 모델
    results = my_hands.process(imgRGB)

    recognized_gesture = None

    # 손 랜드마크 인식 결과 있을 경우.
    if results.multi_hand_landmarks:
        # 손 랜드마크 별로 루프(20개)
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            # 다섯 손가락이 모두 펼쳐져 있는지 아닌지 확인
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
            # 손의 왼쪽/오른쪽 여부 확인
            handedness = results.multi_handedness[idx].classification[0].label
            # 왼손인 경우 엄지손가락 끝이 새끼손가락 끝보다 오른쪽에 있다면 손바닥이 보이는 것으로 판단
            # 오른손인 경우 엄지손가락 끝이 새끼손가락 끝보다 왼쪽에 있다면 손바닥이 보이는 것으로 판단
            palm_visible = (
                (handLms.landmark[4].x > handLms.landmark[20].x)
                if handedness == "Left"
                else (
                    (handLms.landmark[4].x < handLms.landmark[20].x)
                    if handedness == "Right"
                    else False
                )
            )
            # text 위치
            text_x = int(handLms.landmark[0].x * w)
            text_y = int(handLms.landmark[0].y * h)

            for g in gesture:
                if g[:5] == open:
                    gesture_name = g[5]
                    # 손바닥이 보이는 "V"와 손등이 보이는 "Reverse V" 구분
                    if gesture_name == "V" and not palm_visible:
                        gesture_name = "Reverse V"
                    # 화면에 텍스트 표시
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


# Function to provide video stream from webcam
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
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    yield (b"--frame\r\n" b"Content-Type: text/plain\r\n\r\n" + b"WEB_CAM_OFF\r\n")


# Route for the home page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "countries": country_gestures.keys()}
    )


# Route to handle gesture selection and restart webcam
@app.post("/set_gesture", response_class=HTMLResponse)
async def set_gesture(request: Request, country: str = Form(...)):
    global TARGET_GESTURE, webcam_active, webcam_start_time

    TARGET_GESTURE = country_gestures.get(country)

    # Reset the webcam state to restart the gesture recognition process
    webcam_active = True
    webcam_start_time = time.time()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "countries": country_gestures.keys(),
            "selected_country": country,
        },
    )


# Route to stream video feed
@app.get("/video")
def video_feed():
    return StreamingResponse(
        get_stream_video(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# fastapi 구동 명령어: uvicorn main:app --reload
