import numpy as np
import cv2

src = np.ones((500,500),np.uint8)*255
dst = np.ones((500,500),np.uint8)*255

mer = np.zeros((500,500,3),np.uint8)

mer[:,:,0]=src
mer[:,:,1]=src
mer[:,:,2]=dst

cv2.imshow("merged image",mer)
cv2.waitKey(0)

