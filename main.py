from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import mediapipe as mp
import math
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import time
from collections import deque
import asyncio
import uuid
import uvicorn
import random

# FastAPI 앱 초기화
app = FastAPI()

# Jinja2 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")
# 정적 파일 디렉토리 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 손 감지를 위해 MediaPipe 초기화
mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 손가락 위치를 비교하여 손가락이 열려있는지 닫혀있는지 판단하기 위한 인덱스
compareIndex = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
open = [False] * len(compareIndex)  # 각 손가락의 열림/닫힘 상태를 추적하기 위한 변수

# 가능한 제스처와 해당하는 손가락 열림/닫힘 패턴 정의
gesture = [
    [True, True, True, True, True, "Open Plam"],
    [False, False, True, True, True, "OK"],
    [False, False, False, False, True, "Mini"],
    [True, False, False, False, False, "Thumb up"],
    [True, True, False, False, True, "Lock&Roll"],
    [False, True, True, False, False, "V"],
]

# 국가별 대상 제스처 정의
country_gestures = {
    "Brazil": "OK",
    "Turkey": "Thumb up",
    "Middle East": "Lock&Roll",
    "France": "V",
    "Australia": "Mini",
    "Greece": "Lock&Roll",
    "UK": "Mini",
    "New Zealand": "V",
    "Italy": "Reverse V",
    "China": "Mini",
}

# 나라별 제스처 매핑
country_gestures = {
    "Brazil": ["OK"],
    "Turkey": ["OK","V", "Open Palm"],
    "Middle East": ["OK", "Thumb up"],
    "France": ["OK"],
    "Australia": ["Thumb up", "Reverse V"],
    "Greece": ["Thumb up", "V", "Open Palm"],
    "UK": ["Reverse V"],
    "New Zealand": ["Reverse V"],
    "Italy": ["Rock&Roll"],
    "China": ["mini"],
}

# 제스쳐 별 가이드라인 매핑
gesture_guidline = {
    "Open Palm" : "palm.png",
    "OK" : "ok.png",
    "mini" : "mini.png",
    "Thumb up" : "thumb.png",
    "Rock&Roll" : "rocknrole.png",
    "V" : "v.png",
    "Reverse V" : "rv.png"
}

# 세션별 데이터를 저장하기 위한 클래스
class SessionState:
    def __init__(self):
        self.webcam_active = True  # 웹캠 활성화 여부
        self.webcam_start_time = time.time()  # 웹캠 시작 시간
        self.gesture_buffer = deque(maxlen=30)  # 인식된 제스처를 저장하기 위한 버퍼
        self.target_gesture = None  # 세션의 대상 제스처
        self.overlay_image = None # 대상 제스쳐 가이드라인
        self.websocket = None  # WebSocket 연결 저장


# 세션 ID를 사용하여 세션 상태를 저장하는 딕셔너리
sessions = {}


# 두 점 사이의 유클리드 거리를 계산하는 유틸리티 함수
def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2)) + math.sqrt(math.pow(y1 - y2, 2))

# PNG 이미지를 프레임에 오버레이하는 함수
def overlay_image_on_frame(frame, overlay):
    # 이미지 크기 조정 (프레임과 동일한 크기로 조정)
    overlay_resized = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))

    # PNG 파일은 투명한 배경을 가지고 있기 때문에 알파 채널을 고려하여 오버레이합니다.
    alpha_overlay = overlay_resized[:, :, 3] / 255.0  # 알파 채널
    alpha_frame = 1.0 - alpha_overlay

    # 오버레이 이미지의 알파 채널을 고려하여 합성
    for c in range(0, 3):
        frame[:, :, c] = (alpha_overlay * overlay_resized[:, :, c] +
                          alpha_frame * frame[:, :, c])

    return frame

# 각 프레임을 처리하고 제스처를 인식하는 함수
def process_frame(frame, session_state: SessionState):
    h, w, c = frame.shape  # 프레임의 크기 가져오기
    imgRGB = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )  # MediaPipe를 위해 프레임을 RGB로 변환
    results = my_hands.process(imgRGB)  # 손 감지를 위해 프레임 처리

    recognized_gesture = None  # 인식된 제스처를 저장하는 변수

    if results.multi_hand_landmarks:  # 손이 감지된 경우
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            # 각 손가락이 열려있는지 닫혀있는지 판단
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

            # handedness (왼손 또는 오른손) 결정
            handedness = results.multi_handedness[idx].classification[0].label

            # 손바닥이 보이는지 확인 (엄지 손가락 위치를 기준으로)
            # 왼손: 엄지 손가락 끝이 새끼 손가락 끝보다 오른쪽에 있어야 손바닥이 보이는 상태
            # 오른손: 엄지 손가락 끝이 새끼 손가락 끝보다 왼쪽에 있어야 손바닥이 보이는 상태
            palm_visible = (
                (handLms.landmark[4].x > handLms.landmark[20].x)
                if handedness == "Left"
                else (
                    (handLms.landmark[4].x < handLms.landmark[20].x)
                    if handedness == "Right"
                    else False
                )
            )
            # 제스처를 표시하기 위한 텍스트 위치 계산
            text_x = int(handLms.landmark[0].x * w)
            text_y = int(handLms.landmark[0].y * h)

            # 인식된 제스처와 미리 정의된 제스처 비교
            for g in gesture:
                if g[:5] == open:
                    gesture_name = g[5]
                    if gesture_name == "V" and not palm_visible:
                        gesture_name = "Reverse V"
                    # 프레임에 인식된 제스처 표시
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

            # 프레임에 손 랜드마크 그리기
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # 웹캠이 시작된 직후에는 "준비 중..." 메시지 표시
    if (
        session_state.webcam_start_time is None
        or time.time() - session_state.webcam_start_time < 2
    ):
        cv2.putText(
            frame, "Preparing...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        return frame

    frame = overlay_image_on_frame(frame, session_state.overlay_image)

    # 인식된 제스처를 제스처 버퍼에 추가
    session_state.gesture_buffer.append(recognized_gesture)

    # 대상 제스처가 버퍼에서 40% 이상 인식된 경우
    if (
        session_state.gesture_buffer.count(session_state.target_gesture)
        / len(session_state.gesture_buffer)
        > 0.4
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


# 비디오 프레임을 캡처하고 처리하는 비동기 제너레이터 함수
async def get_stream_video(session_id: str):
    session_state = sessions[session_id]
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    if not cap.isOpened():
        print("Cannot open the camera!")
        return

    try:
        while session_state.webcam_active:
            success, frame = cap.read()  # 웹캠에서 프레임 읽기
            if not success or frame is None or frame.size == 0:
                print("Warning: Frame not captured correctly")
                await asyncio.sleep(0.1)
                continue

            frame = process_frame(frame, session_state)  # 프레임 처리
            ret, buffer = cv2.imencode(".jpg", frame)  # 프레임을 JPEG로 인코딩
            frame = buffer.tobytes()

            # WebSocket에 프레임 전송 (연결된 경우)
            if session_state.websocket:
                await session_state.websocket.send_bytes(frame)
            else:
                # WebSocket이 아직 연결되지 않은 경우 프레임 반환
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(0.1)
    finally:
        cap.release()  # 사용이 끝난 웹캠 해제


# 실시간 통신을 위한 WebSocket 엔드포인트
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()  # WebSocket 연결 수락
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    session_state = sessions[session_id]
    session_state.websocket = websocket  # WebSocket 연결 세션에 저장

    try:
        while True:
            data = await websocket.receive_text()  # 클라이언트로부터 데이터 수신
            if data == "get_gesture":
                # 제스처 인식 데이터를 클라이언트로 전송하는 예제
                if session_state.gesture_buffer:
                    current_gesture = session_state.gesture_buffer[-1]
                    await websocket.send_text(
                        current_gesture or "No gesture recognized"
                    )
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        session_state.websocket = None  # WebSocket 연결이 끊긴 경우 처리
        print(f"WebSocket {session_id} disconected.")


# 앱의 메인 페이지, index.html 템플릿으로 렌더링
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session_id = request.cookies.get("session_id")  # 쿠키에서 세션 ID 가져오기
    if not session_id:
        session_id = str(uuid.uuid4())  # 세션 ID가 없는 경우 새로 생성

    if session_id not in sessions:
        sessions[session_id] = SessionState()

    # 세션에 대한 대상 제스처 임의 설정
    target_country = "Brazil"
    gestures = country_gestures.get(target_country)
    sessions[session_id].target_gesture =  random.choice(gestures)
    sessions[session_id].overlay_image = cv2.imread(f"static/{gesture_guidline.get(sessions[session_id].target_gesture)}", cv2.IMREAD_UNCHANGED)

    # index.html 템플릿에 필요한 데이터와 함께 렌더링
    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "countries": country_gestures.keys(),
            "selected_country": "Brazil",
            "session_id": session_id,
            "target_gesture": sessions[session_id].target_gesture,
            "overlay_image" : sessions[session_id].overlay_image
        },
    )
    response.set_cookie(key="session_id", value=session_id)  # 세션 ID 쿠키 설정

    return response


# 세션에 대한 비디오 피드를 제공하는 엔드포인트
@app.get("/video_feed/{session_id}")
async def video_feed(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return StreamingResponse(
        get_stream_video(session_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# 이 스크립트가 직접 실행되는 경우 FastAPI 앱 실행
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# lsof -i :8000
# fastapi 구동 명령어: uvicorn main:app --reload
