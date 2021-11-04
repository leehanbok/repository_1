"""
OpenCV GUI 인터페이스 활용하기
아래 목차에 있는 시간을 참고하여 영상을 학습하시기를 바랍니다.

윈도우 생성하기(00:38~)
이미지 창에 그리기(03:12~)
키보드 활용하기(06:49~)
마우스 활용하기(10:22~)]
ROI(14:34~)  :  그림중 특정 위치만 확인하고자 할때 사용.
"""
import cv2
import time

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        param[0] = cv2.circle(param[0], (x,y), 50, (255,0,0), 5)

if __name__ == "__main__":
    img = cv2.imread("./Lenna.png", cv2.IMREAD_COLOR)
    cv2.namedWindow("My Image", cv2.WINDOW_NORMAL)

    roi = cv2.selectROI("My Image", img)
    # 내가 설정 roi 좌표값을 출력한다.
    print(roi)