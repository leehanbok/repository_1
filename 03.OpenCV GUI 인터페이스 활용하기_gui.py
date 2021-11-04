"""
OpenCV GUI 인터페이스 활용하기
아래 목차에 있는 시간을 참고하여 영상을 학습하시기를 바랍니다.

윈도우 생성하기(00:38~)
이미지 창에 그리기(03:12~)
키보드 활용하기(06:49~)
마우스 활용하기(10:22~)]
ROI(14:34~)
"""
import cv2
import time

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        param[0] = cv2.circle(param[0], (x,y), 50, (255,0,0), 5)

if __name__ == "__main__":
    img = cv2.imread("./Lenna.png", cv2.IMREAD_COLOR)
    cv2.namedWindow("My Image", cv2.WINDOW_AUTOSIZE)

    param = [img]
    cv2.setMouseCallback("My Image", onMouse, param)
    
    # time.sleep(5000)

    while True:
        # 0 : 무한정. key 입력받으면 종료됨.
        key = cv2.waitKey(1000)
        if key == ord("q"):
            break
        elif key == ord("g"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 영상을 gray로 표현

        cv2.imshow("My Image", img)