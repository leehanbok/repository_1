"""
05.이미지 위에 정보 표시하기
아래 목차에 있는 시간을 참고하여 영상을 학습하시기를 바랍니다.

선 그리기(00:36~)
원 그리기(04:09~)
사각형 그리기(07:48~)
텍스트 쓰기(10:08~)
"""
import cv2

if __name__ == "__main__":
    img = cv2.imread("./Lenna.png", cv2.IMREAD_COLOR)
    # 대각선으로 그리기.
    # (255, 0, 0) : 파란색
    # line_img = cv2.line(img, (0,0), (img.shape[0], img.shape[1]), (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.imwrite("./draw_line.jpg", line_img)

    # line_img = cv2.line(img, (0,0), (img.shape[0], img.shape[1]), (255, 0, 0), 1, cv2.LINE_AA)
    # 원 그리기
    # 행 이 x 값임.  x : shape의 컬럼. y : shape의 
    # circle_img = cv2.circle(img, (img.shape[1]//2, img.shape[0]//2), 100, (0,255,0), 5)
    # cv2.imwrite("./draw_circle.jpg", circle_img)

    # recgtangle 그리기
    # 행 이 x 값임.  x : shape의 컬럼. y : shape의 
    # rectangle_img = cv2.rectangle(img, (100, 100), (300, 400), (0,0,255), 5)
    # cv2.imwrite("./draw_rectangle.jpg", rectangle_img)

    # 텍스트 넣기  (단, 한글은 안됨.)
    putText_img = cv2.putText(img, "This is Lenna", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.7, (128,0,128), 5)
    # putText_img = cv2.putText(img, "한글", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.7, (128,0,128), 5)  ## ??? 로 표현됨.
    cv2.imwrite("./draw_putText.jpg", putText_img)