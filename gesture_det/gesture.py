import cv2
import mediapipe as mp
import math
import sys


def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2)) + math.sqrt(math.pow(y1 - y2, 2))


mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Existing compareIndex remains the same
compareIndex = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

# 손가락이 펴져 있는지 확인
open = [False] * len(compareIndex)

# 각 제스쳐별 다섯손가락 상태(펴져 있는지 유무), 제스쳐명 정의
gesture = [
    [True, True, True, True, True, "Open Parm"],
    [False, False, True, True, True, "OK"],
    [False, False, False, False, True, "mini"],
    [True, False, False, False, False, "Thumb up"],
    [True, True, False, False, True, "Rock&Roll"],
    [True, True, False, False, True, "Rock&Roll"],
    [False, True, True, False, False, "V"],
]

# 카메라 열기q
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라가 열려있지 않습니다!")  # 열리지 않았으면 문자열 출력
    sys.exit()

while True:
    # image 캡쳐
    success, img = cap.read()
    h, w, c = img.shape
    # RGR -> RGB 색상 변경
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 손 검출 및 분석
    results = my_hands.process(imgRGB)

    # 손 랜드마크가 존재하는 경우
    if results.multi_hand_landmarks:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            # 각 손가락의 상태를 업데이트
            for i in range(0, 5):
                # 손목과 손가락의 특정 랜드마크 사이의 거리 계산
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
            # 손가락 상태 출력
            print(open)

            # 왼손인지 오른손인지 여부
            handedness = results.multi_handedness[idx].classification[0].label

            # 왼손인 경우
            if handedness == "Left":
                # 엄지손가락 좌표가 새끼손가락 좌표보다 클 때 손바닥 보이는 상태
                palm_visible = handLms.landmark[4].x > handLms.landmark[20].x
            elif handedness == "Right":
                # 엄지손가락 좌표가 새끼손가락 좌표보다 작을 때 손바닥 보이는 상태
                palm_visible = handLms.landmark[4].x < handLms.landmark[20].x
            else:
                palm_visible = False

            text_x = handLms.landmark[0].x * w
            text_y = handLms.landmark[0].y * h
            for i in range(0, len(gesture)):
                flag = True
                for j in range(0, 5):
                    if gesture[i][j] != open[j]:
                        flag = False

                if flag == True:
                    # 제스쳐가 V인 경우
                    if gesture[i][5] == "V":
                        # 손바닥이 보이지 않을 때
                        if not palm_visible:
                            gesture_name = "Reverse V"
                        # 손바닥이 보일 때
                        else:
                            gesture_name = "V"
                    # 그 이외에 제스처인 경우
                    else:
                        gesture_name = gesture[i][5]

                    cv2.putText(
                        img,
                        gesture_name,
                        (round(text_x) - 50, round(text_y) - 250),
                        cv2.FONT_HERSHEY_PLAIN,
                        4,
                        (0, 0, 0),
                        4,
                    )

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("HandTracking", img)
    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
