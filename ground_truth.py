import cv2
import os
import glob
from pathlib import Path


path = './gt/*.*'
dir = './gt/'
file_list = glob.glob(path)

images=[]

text=''

txt=''

for name in file_list:
    names=name.split('\\')

    if name[len(name)-1] == 'g':

        img = cv2.imread(str(name))
        h, w, c = img.shape

        continue

    txt = open(str(name), 'r')
    line = txt.readline()

    while line:
        print(line)
        list = line.split(' ')
        if list[0] == str(0):
            text = "pedestrian"
        elif list[0] == str(1):
            text = "Bollard"
        elif list[0] == str(2):
            text = "Bicycle"

        #[1]left  [2]center  list[3]가로길이 iist[4]세로길이
        center = h * float(list[2])
        width = w * float(list[3])
        height = h * float(list[4])

        left = str(int(float(list[1]) * float(w)))
        top = str(int(float(center) - float(height)/2))
        right = str(int((float(list[1]) * float(w)) + width))
        bottom = str(int(float(center) + float(height)/2))

        text = text +' '+ left+' '+top+' '+right+' '+bottom+'\n'
        # left  center width height
        newfile = open('./gt/newfile/'+names[1],'a')
        newfile.writelines(text)
        newfile.close()

        line = txt.readline()
    txt.close()




