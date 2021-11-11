import os
import glob
import cv2
import numpy
import xml.etree.ElementTree as ET
import tqdm

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 
               'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 
               'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 
               'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

classes = list(classes_num.keys())


def voc_load_data(img_dir_path, annotation_path, batch=10):
    images, labels = [], []
    img_file_list = glob.glob((img_dir_path + "/*.jpg"))

    for i in range(len(img_file_list)):
        for img_path in tqdm.tqdm(img_file_list[batch * i: batch * (i + 1)]):

            # Read image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_h, image_w = image.shape[0:2]

            # Resize (x, y)크기 이미지 -> (488, 488)로 변경
            image = cv2.resize(image, (448, 448))
            # normalization 이미지 데이터를 0~1 사이의 값으로 변경
            image = image / 255.0  ## 표준화 진행

            images.append(image)

            # Read xml file  :: 이미지파일과 XML파일명이 대응된다는 전제.
            # --> 경로제거하고 파일명만 추출..(아래2줄)
            xml_name = os.path.split(img_path)[-1]
            xml_name = xml_name.split(".")[-2]
            xml_path = annotation_path + f"/{xml_name}.xml"

            # parse xml 선언
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Empty matrix, (7, 7, 25)크기의 0으로 채워진 label_matrix를 만드세요 
            ## 임의의 정답BOX를 구성한다.
            ## YOLO 에서는 앵커BOX가 2개 필요.(그래서 7,7,30(
            ## 하지만, 이번실습에서는 정답이 1개(앵커BOX1개면됨)로 고정하기로 함.(difficult 0), class 20 개?
            label_matrix = numpy.zeros((7,7,25))

            ## 섹션(ex. <> 쌓여있는거) 모두 reading.
                # <object>
                # 	<name>dog</name>
                # 	<pose>Left</pose>
                # 	<truncated>1</truncated>
                # 	<difficult>0</difficult>
                # 	<bndbox>
                # 		<xmin>48</xmin>
                # 		<ymin>240</ymin>
                # 		<xmax>195</xmax>
                # 		<ymax>371</ymax>
                # 	</bndbox>
                # </object>
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                class_name = obj.find('name').text
                
                ## class 명이 우리가 선언한 classes에 없으면 패스,
                ## 또한 difficult 0 이 아니어도 패스 하자..
                if class_name not in classes or difficult == "1":
                    continue

                # Set class id
                cls_id = classes.index(class_name)
                xmlbox = obj.find('bndbox')
                tlx, tly = int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text)
                brx, bry = int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)

                # x좌표, y좌표를 0~1 사이의 값으로 일반화 하세요.
                ## 셀안에서의 중심좌표..즉, 정규화된(1*1) 된 좌표가 구해짐.
                x = (brx + tlx) / 2 / image_w
                y = (bry + tly) / 2 / image_h
                
                # w, h를 0~1 사이의 값으로 일반화 하세요
                # # (brx - tlx) = 가로길이
                w = (brx - tlx) / image_w
                h = (bry - tly) / image_h

                # (7x7)그리드 셀의 좌표와 셀 안에서의 좌표를 구하세요.
                loc = [7 * x, 7 * y]
                # x,y 를 뒤집(??)어서 intger 처리하면 그리드 셀 안에서의 좌표가 나온다고 함. 
                loc_i = int(loc[1]) # 그리드 좌표 y --> 즉, integer casting하면 셀number값이 구해짐.
                loc_j = int(loc[0]) # 그리드 좌표 x
                y = loc[1] - loc_i # 그리드 셀 안에서의 y좌표
                x = loc[0] - loc_j # 그리드 셀 안에서의 x좌표

                if label_matrix[loc_i, loc_j, 24] == 0:
                    # [<----------20---------->|x|y|w|h|pc]
                    label_matrix[loc_i, loc_j, cls_id] = 1
                    label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                    label_matrix[loc_i, loc_j, 24] = 1  # response ==>> PC 값임. (무조건 정답이야)

            labels.append(label_matrix)

        return numpy.array(images), numpy.array(labels)