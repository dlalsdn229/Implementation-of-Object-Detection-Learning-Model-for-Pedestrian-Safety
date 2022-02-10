import csv
import os
import shutil
import cv2

path = "E:/DR detection datasets/train/"

d_path0 = 'E:/DR detection datasets/No DR/'
d_path1 = 'E:/DR detection datasets/Mild/'
d_path2 = 'E:/DR detection datasets/Moderate/'
d_path3 = 'E:/DR detection datasets/Severe/'
d_path4 = 'E:/DR detection datasets/Proliferative DR/'

#filesArray = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path,x))]

f = open('trainLabels.csv','r',encoding='utf-8')
reader = csv.reader(f)

path0 = "E:/DR detection datasets/trainset/label5/No DR400"
path1 = "E:/DR detection datasets/trainset/label5/Mild400"
path2 = "E:/DR detection datasets/trainset/label5/Moderate400"
path3 = "E:/DR detection datasets/trainset/label5/Servere400"
path4 = "E:/DR detection datasets/trainset/label5/Proliferative DR400"
if not os.path.exists(path0):
    os.mkdir(path0)
if not os.path.exists(path1):
    os.mkdir(path1)
if not os.path.exists(path2):
    os.mkdir(path2)
if not os.path.exists(path3):
    os.mkdir(path3)
if not os.path.exists(path4):
    os.mkdir(path4)



for idx, value in enumerate(reader):
    #print(idx,value[0],value[1])
    img = cv2.imread(path+value[0]+".jpeg")
    #cv2.imshow('',img)
    if value[1] == '0':
        cv2.imwrite(d_path0 + value[0]+".jpeg",img)        
                     
    elif value[1] == '1':
        cv2.imwrite(d_path1 + value[0]+".jpeg",img)

    elif value[1] == '2':
        cv2.imwrite(d_path2 + value[0]+".jpeg",img)

    elif value[1] == '3':
        cv2.imwrite(d_path3 + value[0]+".jpeg",img)

    elif value[1] == '4':
        cv2.imwrite(d_path4 + value[0]+".jpeg",img)
            
    if idx % 1000 == 0:
        print(str(idx)+"/35126")

'''
for image_name, label in enumerate(reader):
    print(image_name,label[0],label[1])
    
categories = ['No DR', 'Mild','Moderate', 'Severe', 'Proliferative DR']


'''
