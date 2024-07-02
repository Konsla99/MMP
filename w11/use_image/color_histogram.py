
#pg 14 histogram

import cv2
from matplotlib import pyplot as plt
img_path = 'D:/code/mmp/W10/w10_img/low.jpg'
img = cv2.imread(img_path)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()