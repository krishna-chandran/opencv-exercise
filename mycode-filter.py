import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(r'football.mp4')
# cap = cv2.VideoCapture(0)


i = 0
while(1):
    ret, frame = cap.read()

    blurr = cv2.GaussianBlur(frame, (5, 5), 0)
    imgG = cv2.cvtColor(blurr, cv2.COLOR_BGR2GRAY)
    imgC = cv2.Canny(imgG, 50, 90)
    imgM = cv2.morphologyEx(imgC, cv2.MORPH_CLOSE, (5,5))
    (cont, _) = cv2.findContours(imgM.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    

    frame = cv2.resize(imgM, (960, 540))

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()