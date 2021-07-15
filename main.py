import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import mediapipe

#WORK ON VIDEO
#
# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(cv2.CAP_PROP_FPS,60)
# segmentor=SelfiSegmentation()
# fpsreader=cvzone.FPS()
# #for single imgae backgroun changer
# #imgbg=cv2.imread("images/EAM_Nuvolari_S1_640x480.jpg")
#
# list=os.listdir("images")
# imglist=[]
# for imgpath in list:
#     img=cv2.imread(f'images/{imgpath}')
#     imglist.append(img)
# indeximg=0
# while True:
#     success,img=cap.read()
#     imgout=segmentor.removeBG(img,imglist[indeximg],threshold=0.8)
#     imgstacked=cvzone.stackImages(([img,imgout]),2,1)
#
#     fps,imgstacked=fpsreader.update(imgstacked)
#     cv2.imshow("image", imgstacked)
#
#     key=cv2.waitKey(1)
#     if key== ord('a'):
#         if indeximg>0:
#             indeximg-=1
#     elif key==ord('d'):
#         if indeximg<len(list)-1:
#             indeximg+=1
#     elif key==ord('q'):
#         break



#WORK ON IMAGES
segmentor=SelfiSegmentation()
fpsreader=cvzone.FPS()

# remove background
# img1=cv2.imread("view1.jpeg")
# img1=cv2.resize(img1,(700,500))
# imgout=segmentor.removeBG(img1,(255,0,255),threshold=0.1)
# cv2.imshow("image1",imgout)
# cv2.waitKey(0)

#change background
img1=cv2.imread("view1.jpeg")
img1=cv2.resize(img1,(640,480))
img2=cv2.imread("images/P_Vega_de_Calardos_(640x480).jpg")
imgout=segmentor.removeBG(img1,img2,threshold=0.1)
imgstacked=cvzone.stackImages(([img1,imgout]),2,1)
fps,imgstacked=fpsreader.update(imgstacked)
cv2.imshow("image", imgstacked)
cv2.waitKey(0)

