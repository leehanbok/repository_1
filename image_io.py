
import cv2

if __name__ == "__main__":
    # print("Hello world")
    # print(cv2.__version__)  ## 4.5.4-dev

    ## TEST1: imagae 읽어오기
    # color_img = cv2.imread("./Lenna.png", cv2.IMREAD_COLOR)
    # gray_img = cv2.imread("./Lenna.png", cv2.IMREAD_GRAYSCALE)

    # print(color_img.shape)  # (512, 512, 3)
    # print(gray_img.shape)   # (512, 512)

    ## TEST2 : imagae 저장하기
    # gray_img = cv2.imread("./Lenna.png", cv2.IMREAD_GRAYSCALE)
    # if cv2.imwrite("./lenna.jpg", gray_img):
    #     print("저장완료")
    # else:
    #     print("저장실패")

    # TEST3 : cvtColor() 사용해서 회색으로 저장
    # color_img = cv2.imread("./Lenna.png", cv2.IMREAD_COLOR)
    # gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  ## BlueRedGreen

    # cv2.imwrite("./converted_to_gray.jpg", gray_img)

    # TEST4 : resize
    # color_img = cv2.imread("./Lenna.png", cv2.IMREAD_COLOR)
    # # resize_img = cv2.resize(color_img, (1024, 1024))
    # resize_img = cv2.resize(color_img, (128, 128))
    # cv2.imwrite("./resized_img.jpg", resize_img)

    # TEST5 : 특정구간만 chrop 자르기
    color_img = cv2.imread("./Lenna.png", cv2.IMREAD_COLOR)
    cropped_img = color_img[0:color_img.shape[0] // 2, 0:color_img.shape[1] // 2]
    print(cropped_img.shape)  ## (256, 256, 3)
    cv2.imwrite("./croppped_img.jpg", cropped_img)

