from curses import window
import os
from bs4 import BeautifulSoup
import cv2
import matplotlib.pyplot as plt
from customDataset import CustomDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Get the length of the training data(from /Main/train.txt) to split the image collection(JPEGImages) based on that.
arr = []
for data in os.listdir("/y/ayhassen/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007//ImageSets/Main"):
    with open(os.path.join("/y/ayhassen/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007//ImageSets/Main", data), 'r') as f:
        text = f.read()
        arr.append(len(text))

data = []
labels = []
bboxes = []
image_paths = []



# Parse your xml annotation files into a format your Custom Dataset can work with. Basically, get your data from the different files and folders then feed it to the Dataset. 
trainData = arr[1]
# print(trainData)
trainlength = 0
for filename in os.listdir("/y/ayhassen/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/Annotations"):
    if trainlength <= trainData:
        with open(os.path.join("/y/ayhassen/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/Annotations", filename), 'r') as f:
            text = f.read()
            trainlength+=1
            train_data = BeautifulSoup(text, "xml")
            image = train_data.find('filename')
            image = str(image)
            image_name = image[10:-11]
            # print(image_name)
            image_path = os.path.join("/y/ayhassen/pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/JPEGImages/", image_name)
            label = train_data.find('object')
            label = label.find('name')
            label = str(label)[6: -7]
            bbox = train_data.find('object')
            bbox = bbox.find('bndbox')
            bbox1 = bbox.find('xmin')
            bbox1 = str(bbox1)
            bbox1 = bbox1[6:-7]
            bbox2 = bbox.find('ymin')
            bbox2 = str(bbox2)
            bbox2 = bbox2[6:-7]
            bbox3 = bbox.find('xmax')
            bbox3 = str(bbox3)
            bbox3 = bbox3[6:-7]
            bbox4 = bbox.find('ymax')
            bbox4 = str(bbox4)
            bbox4 = bbox4[6:-7]

            image = cv2.imread(image_path)
            image = cv2.resize(image, (224,224))
            data.append(image)
            labels.append(label)
            bboxes.append((bbox1, bbox2, bbox3, bbox4))
            image_paths.append(image_path)

            print(trainlength)
            # print(data[0])
            # print(labels[0])
            # print(bboxes[0])
            # print(image_paths[0])
            # print(type(image))
            # cv2.rectangle(image, (int(bbox1), int(bbox2)), (int(bbox3), int(bbox4)), (0, 0, 0), 1)
            # status = cv2.imwrite("/home/ayhassen/starter_project/images/test.png", image)
            # print(status)


data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
image_paths = np.array(image_paths)
le = LabelEncoder()
labels = le.fit_transform(labels)


def export_data():
    return [data, labels, bboxes, image_paths]





