import cv2
import math
import time
from collections import deque
import asyncio
import uuid
import random

import cv2
import mediapipe as mp

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from websockets.exceptions import ConnectionClosed

# CORS 설정: 모든 출처에서의 요청 허용
origins = ["*"]

# FastAPI 앱 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# MediaPipe 손 감지 초기화
mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 제스처 인식 패턴
compareIndex = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
# 손가락 상태(펼침/접힘
open = [False] * len(compareIndex)

# 제스처 목록(손가락 상태, 제스처 이름)
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
    "Brazil": ["OK"],
    "Turkiye": ["OK", "V", "Open Palm"],
    "Saudi": ["OK", "Thumb up"],
    "France": ["OK"],
    "Australia": ["Thumb up", "Reverse V"],
    "Greece": ["Thumb up", "V", "Open Palm"],
    "England": ["Reverse V"],
    "NewZealand": ["Reverse V"],
    "Italia": ["Rock&Roll"],
    "China": ["mini"],
}

# 제스쳐 별 가이드라인 매핑
gesture_guidline = {
    "Open Palm": "palm.png",
    "OK": "ok.png",
    "mini": "mini.png",
    "Thumb up": "thumb.png",
    "Rock&Roll": "rocknroll.png",
    "V": "v.png",
    "Reverse V": "rv.png",
}


# 세션 상태 클래스
class SessionState:
    def __init__(self):
        self.webcam_active = True
        self.webcam_start_time = time.time()
        self.gesture_buffer = deque(maxlen=30)  # 제스처 버퍼
        self.country = None
        self.target_gesture = None
        self.websocket = None
        self.gesture_recognized = False
        self.overlay_image = None


sessions = {}


# 유클리드 거리 계산 함수
def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# PNG 이미지를 프레임에 오버레이하는 함수
def overlay_image_on_frame(frame, overlay):
    # 이미지 크기 조정 (프레임과 동일한 크기로 조정)
    overlay_resized = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))

    # PNG 파일은 투명한 배경을 가지고 있기 때문에 알파 채널을 고려하여 오버레이합니다.
    alpha_overlay = overlay_resized[:, :, 3] / 255.0  # 알파 채널
    alpha_frame = 1.0 - alpha_overlay

    # 오버레이 이미지의 알파 채널을 고려하여 합성
    for c in range(0, 3):
        frame[:, :, c] = (
            alpha_overlay * overlay_resized[:, :, c] + alpha_frame * frame[:, :, c]
        )

    return frame


# 프레임 처리 함수
def process_frame(frame, session_state: SessionState):
    h, w, _ = frame.shape
    # RGR -> RGB 색상 변경
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 손 감지 프로세싱
    results = my_hands.process(imgRGB)

    recognized_gesture = None
    # 손 감지가 되었을 때
    if results.multi_hand_landmarks:
        # 랜드마크(20개) 별 결과 처리
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            # 특정랜드마크와 비교하여 손가락 상태 판별
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
            # 왼손/오른손 판별
            handedness = results.multi_handedness[idx].classification[0].label
            if handedness == "Left":
                # 엄지손가락 좌표가 새끼손가락 좌표보다 클 때 손바닥 보이는 상태(True)
                palm_visible = handLms.landmark[4].x > handLms.landmark[20].x
            elif handedness == "Right":
                # 엄지손가락 좌표가 새끼손가락 좌표보다 작을 때 손바닥 보이는 상태(True)
                palm_visible = handLms.landmark[4].x < handLms.landmark[20].x
            # 웹캠 내 텍스트 출력 위치
            text_x = int(handLms.landmark[0].x * w)
            text_y = int(handLms.landmark[0].y * h)
            # 제스처 인식
            for g in gesture:
                if g[:5] == open:
                    gesture_name = g[5]
                    # 제스처가 V일 때 손바닥이 보이지 않으면 Reverse V
                    if gesture_name == "V" and not palm_visible:
                        gesture_name = "Reverse V"
                    # 제스처 명 웹캠 내 텍스트 출력
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
    # 웹캠 시작 시간이 없거나 2초 이내일 때
    if (
        session_state.webcam_start_time is None
        or time.time() - session_state.webcam_start_time < 2
    ):
        # 웹캠 준비 중 텍스트 출력
        cv2.putText(
            frame, "Preparing...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        # return frame
    if session_state.overlay_image is not None:
        frame = overlay_image_on_frame(frame, session_state.overlay_image)
    # 제스처 인식 결과가 있을 때 버퍼 리스트에 추가
    if recognized_gesture is not None:
        session_state.gesture_buffer.append(recognized_gesture)

    # 버퍼 리스트에 타겟 제스처가 60% 이상일 때
    if len(session_state.gesture_buffer) > 10:  # Ensure buffer is large enough
        if (
            session_state.gesture_buffer.count(session_state.target_gesture)
            / len(session_state.gesture_buffer)
            > 0.6
        ):
            cv2.putText(
                frame,
                f"Target Gesture '{session_state.target_gesture}' Recognized!",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    return frame


# WebSocket을 통해 비디오 스트림 제공
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    session_state = sessions[session_id]
    session_state.websocket = websocket

    # 웹캠 비디오 스트림 제공
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open the camera!")
        return

    try:
        while session_state.webcam_active:
            success, frame = cap.read()
            # 웹캠 좌우 반전
            frame = cv2.flip(frame, 2)
            # 프레임이 없거나 비어있을 때 0.1초 대기
            if not success or frame is None or frame.size == 0:
                await asyncio.sleep(0.1)
                continue
            # 프레임 처리
            frame = process_frame(frame, session_state)
            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                # 비디오 프레임을 바이트로 변환하여 웹소켓 전송
                await websocket.send_bytes(buffer.tobytes())
            # 제스처 인식 결과가 있을 때
            if (
                len(session_state.gesture_buffer) > 10
                and session_state.gesture_buffer.count(session_state.target_gesture)
                / len(session_state.gesture_buffer)
                > 0.6
            ):
                # 제스처 인식 결과 전송
                session_state.gesture_recognized = True
                await websocket.send_text("gesture_recognized")
                break
            await asyncio.sleep(0.03)
    # 웹소켓 연결 종료 시
    except (WebSocketDisconnect, ConnectionClosed):
        session_state.websocket = None
        print(f"WebSocket {session_id} disconnected.")
    finally:
        cap.release()


# 메인 페이지
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session_id = request.cookies.get("session_id")
    # 쿠키에 세션 ID가 없을 때 새로 생성
    if not session_id:
        session_id = str(uuid.uuid4())
    # 세션 상태가 없을 때 새로 생성
    if session_id not in sessions:
        sessions[session_id] = SessionState()

    response = templates.TemplateResponse(
        "Main.html",
        {
            "request": request,
            "countries": country_gestures.keys(),
            "session_id": session_id,
        },
    )
    response.set_cookie(key="session_id", value=session_id)

    return response


# Global 페이지
@app.get("/global", response_class=HTMLResponse)
async def global_page(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    response = templates.TemplateResponse(
        "Global.html", {"request": request, "session_id": session_id}
    )
    response.set_cookie(key="session_id", value=session_id)
    return response


# Situation1 페이지
@app.get("/situation1", response_class=HTMLResponse)
async def situation1(request: Request, country: str = None):
    session_id = request.cookies.get("session_id")
    print(f"Session ID in /situation1: {session_id}")
    # 쿠키에 세션 ID가 없을 때 새로 생성
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"New Session ID generated: {session_id}")
    # 세션 상태가 없을 때 새로 생성
    if session_id not in sessions:
        sessions[session_id] = SessionState()
        print(f"New session state created for Session ID: {session_id}")

    # 선택된 국가가 없을 때 Global 페이지로 리다이렉트
    if country is None:
        return RedirectResponse(url="/global")
    elif country == "random":
        country = random.choice(list(country_gestures.keys()))
    sessions[session_id].country = country
    # gestures = country_gestures.get(country)
    response = templates.TemplateResponse(
        "situation1.html",
        {
            "request": request,
            "countries": country_gestures.keys(),
            "selected_country": country,
            "country": sessions[session_id].country,
            "session_id": session_id,
            "target_gesture": sessions[session_id].target_gesture,
            "overlay_image": sessions[session_id].overlay_image,
        },
    )
    response.set_cookie(key="session_id", value=session_id)
    print(f"Set Session ID in cookie: {session_id}")
    return response


# Situation2 페이지
@app.get("/situation2", response_class=HTMLResponse)
async def situation2(request: Request, country: str = None):
    session_id = request.cookies.get("session_id")
    # 쿠키에 세션 ID가 없을 때 새로 생성
    if not session_id:
        session_id = str(uuid.uuid4())
    # 세션 상태가 없을 때 새로 생성
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    country = sessions[session_id].country

    sessions[session_id].target_gesture = random.choice(country_gestures.get(country))
    sessions[session_id].overlay_image = cv2.imread(
        f"app/static/img/gestureGuide/{gesture_guidline.get(sessions[session_id].target_gesture)}",
        cv2.IMREAD_UNCHANGED,
    )

    response = templates.TemplateResponse(
        "situation2.html",
        {
            "request": request,
            "countries": country_gestures.keys(),
            "country": sessions[session_id].country,
            "selected_country": country,
            "session_id": session_id,
            "target_gesture": sessions[session_id].target_gesture,
        },
    )
    response.set_cookie(key="session_id", value=session_id)

    return response


# Success 페이지
@app.get("/success", response_class=HTMLResponse)
async def success(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return RedirectResponse(url="/")

    session_state = sessions[session_id]

    # 세션 상태 초기화
    session_state.gesture_recognized = False

    return templates.TemplateResponse("success.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
