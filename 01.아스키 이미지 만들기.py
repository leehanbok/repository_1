"""
양자화는 영상이 얼마나 섬세하게 색을 표현할 수 있는지 결정합니다.

이번 문제에서는 영상을 흑백으로 바꾸어보고 주어진 레벨에 맞게 양자화를 진행합니다. 
그리고 명도에 따라 아스키 문자를 할당하여 콘솔로 영상을 출력할 수 있는 간단한 아스키 영상을 만들어봅시다.

아래 함수 정의를 참고하여, 문제를 해결합니다.
----------------------------------------------------------------------------------------------------------
함수 정의
1. img2ascii(img, L, ascii_string)
 -. img : OpenCV로 읽은 이미지
 -. L : 양자화 레벨, 2<=L<=162 <= L <= 162<=L<=16 사이의 정수
 -. ascii_string : 실습에서 주어지는 명도를 대신 표현할 문자열
 -. 반환값 : numpy.chararray 타입의 문자로 표현된 이미지

2. print_ascii_img(ascii_img)
 -. 기능 : 파라미터로 받은 ascii_img를 출력해주는 함수
 -. ascii_img : 문자로 이루어진 numpy.chararray 타입 이미지
 -. 반환값 : 없음
-----------------------------------------------------------------------
지시사항
예시 영상을 읽고 흑백 영상으로 바꿉니다.

주어진 양자화 레벨 ‘L’에 맞게 양자화를 진행합니다.

양자화된 명도 ‘i’를 주어진 문자의 인덱스에 해당하는 문자로 치환하세요.

치환된 이미지를 print()로 출력하여 결과를 확인합니다.

Tips!
img2ascii() 함수에서 변수 L을 이용해 양자화 후 ascii_string의 인덱스에 해당하는 문자로 치환하면 됩니다.
"""
import numpy
import cv2

# 아스키 이미지를 출력하는 함수입니다.
def print_ascii_img(ascii_img):
    for  i in range(ascii_img.shape[0]):
        for  j in range(ascii_img.shape[1]):
            print(ascii_img[i, j].decode("utf-8"), end="")
        print()

# 이미지를 아스키로 변환하는 함수입니다.
def img2ascii(img, L, ascii_string):
    ## AttributeError: 'NoneType' object has no attribute 'shape'
    ascii_img = numpy.chararray(img.shape)
    
    # 0 ~ 255 L = 256 -> L = 16
    # 0 16 32 48 64 -> 16 단위.
    qunit = 256 // L   # // 자동으로 정수단위로 변환.

    # 양자화를 수행하고 ascii 문자로 명도를 표현하세요.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 알맞은 명도값 추출..( ex. img[i, j] )
            # img[i, j, 1]  : 회색임으로 채널이 1 임.   0,1,2 (컬러로표시)
            index = img[i, j] // qunit
            ascii_img[i, j] = ascii_string[index]  ## 이미지임에도 콘솔로 출력하기위 함.
    
    return ascii_img

if __name__ == "__main__":
    ascii_string = "@#BPDQOUo=+*~-`."
    
    # 주어진 이미지를 읽어옵니다.
    # 컬러이미지를 회색이미지로 읽기..
    img = cv2.imread("./lena2.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 함수를 호출하여 결과를 확인합니다.
    ascii_img = img2ascii(img, 16, ascii_string)
    print_ascii_img(ascii_img)
