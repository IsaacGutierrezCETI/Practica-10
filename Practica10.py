#Isaac Alejandro Gutiérrez Huerta 19110198 7E1
#Sistemas de Visión Artificial

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Aguila2.png')
roi = cv2.selectROI('roi',img)
recRoi = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

cv2.imwrite('roi.jpg',recRoi)

img2 = cv2.imread('roi.jpg')
imgEsq = img2

mask = np.zeros(img2.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (0,0,300,300)

cv2.grabCut(img2,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img2 = img2*mask2[:,:,np.newaxis]

gray = cv2.cvtColor(imgEsq,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(imgEsq,(x,y),3,255,-1)
    
plt.imshow(img2)
#plt.colorbar()
plt.show()

cv2.imshow('Esquinas',imgEsq)

cv2.waitKey(0)
cv2.destroyAllWindows()
