#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)   # 0 for primary camera 


while True:
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow('frame', frame)
  cv2.imshow('gray', gray)
    

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


cap.release()


cv2.destroyWindow()


# In[ ]:


img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)


# In[1]:


import numpy as np
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
cv2.line(img, (0, 0), (150, 150), (0, 255, 0), 15)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


import numpy as np
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

cv2.rectangle(img, (15, 25), (200, 150), (0, 255, 0), 5)
cv2.circle(img, (100, 63), 55, (0, 0, 255), -1)   # center Coordinates, radius, color, -1 = fill

pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
# pts = pts.reshape((-1, 1, 2))  reshape the pts to 1, 2 dimension as required by polylines function
cv2.polylines(img, [pts], True, (0, 255, 255), 5)    # True for closing the end point of polygon

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


# Text on image

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV!', (0,130), font, 1, (200, 255, 255), 5, cv2.LINE_AA)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


import numpy as np
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

img[55, 55] = [203, 192, 255]
px = img[55, 55]
print(px)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


# ROI = Region of Image

img[100:150, 100:150] = [255, 255, 255]

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:




face = img[37:111, 107:194]
img[0:74, 0:87] = face                 # 111 - 37 = 74, 194 - 107 = 87

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


# Image Arithmatics

import cv2
import numpy as np

img1 = cv2.imread('im1.png')
img2 = cv2.imread('im2.png')

add = img1 + img2

cv2.imshow('add', add)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


# Add all the pixels together

add = cv2.add(img1, img2)

cv2.imshow('add', add)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


# Weighted
import cv2
weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

cv2.imshow('weighted', weighted)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


# Masking

import cv2
import numpy as np

img1 = cv2.imread('im1.png')
img2 = cv2.imread('im2.png')

rows, cols, channels = img2.shape

roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)   # set a threshold of 255, above that pixel value will be 255(white)

cv2.imshow('mask', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


import cv2
import numpy as np

img1 = cv2.imread('im1.png')
img2 = cv2.imread('im2.png')

rows, cols, channels = img2.shape

roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)   # set a threshold of 255, above that pixel value will be 255(white)

mask_inv = cv2.bitwise_not(mask)   # Inverse of mask

img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
img2_fg = cv2.bitwise_and(img2, img2, mask = mask)

dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('img1_bg', img1_bg)
cv2.imshow('img2_fg', img2_fg)
cv2.imshow('dst', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


import cv2
import numpy as np

img = cv2.imread('im3.jpg')
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
retval2, otsu = cv2.threshold(grayscaled, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('original', img)
cv2.imshow('threshold', threshold)
cv2.imshow('threshold2', threshold2)
cv2.imshow('gaus', gaus)
cv2.imshow('otsu', otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


# Video HUE saturation

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # hsv = hue sat val
    lower_red = np.array([150, 150, 50])     # Change these values for other effects
    upper_red = np.array([180, 255, 150])    # Change these values for other effects
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    
    kernel = np.ones((15, 15), np.float32) / 255
    smoothed = cv2.filter2D(res, -1, kernel)            # Smoothening
    
    blur = cv2.GaussianBlur(res, (15, 15), 0)          # Gaussian blur
    median = cv2.medianBlur(res, 15)                # Median Blur
    bilateral = cv2.bilateralFilter(res, 15, 75, 75)
    
    
    
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    #cv2.imshow('smoothed', smoothed)
    cv2.imshow('blur', blur)
    cv2.imshow('median', median)
    cv2.imshow('bilateral', bilateral)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()


# In[1]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # hsv = hue sat val
    lower_red = np.array([150, 150, 50])     # Change these values for other effects
    upper_red = np.array([180, 255, 150])    # Change these values for other effects
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask = mask)
    
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = 1)
    dilation = cv2.dilate(mask, kernel, iterations = 1)
    
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Removes false positive from image
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # removes false negative from image
    
    
    # It is the difference between input image and opening of the image
    cv2.imshow('Tophat', tophat)
    
    # It is the difference between th eclosing of the input image and the input image
    cv2.imshow('Blackhat', blackhat)
    
    cv2.imshow('frame', frame)
    
    cv2.imshow('res', res)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dialtion', dilation)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()


# In[1]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    soblelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 5)
    soblely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
    edges = cv2.Canny(frame, 100, 200)
    
    cv2.imshow('original', frame)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('soblelx', soblelx)
    cv2.imshow('soblely', soblely)
    cv2.imshow('edges', edges)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()


# In[3]:


# Template Matching


import cv2
import numpy as np

img_rgb = cv2.imread('im4.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('im5.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8                                                         # Vary the threshold
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


# Foreground Extraction

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('im6.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect =  (161, 79, 150, 150)     # Vary these values according to the images

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()


# In[1]:


# Corner detection

import cv2
import numpy as np

img = cv2.imread('im7.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)
    
cv2.imshow('Corner', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


# Feature Matching Brute Force

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('im8.jpg', 0)
img2 = cv2.imread('im9.jpg', 0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = 2)
plt.imshow(img3)
plt.show()


# In[ ]:





# In[ ]:





# In[1]:


# Background Reduction
import numpy as np
import cv2


# In[3]:


cap = cv2.VideoCapture('v1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    
    fgmask = fgbg.apply(frame)
    
    cv2.imshow('fgmask', frame)
    cv2.imshow('frame', fgmask)
    
    k = cv2.waitKey(30) & 0xffq
    if k == 27:
        break
        
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[2]:


# Haar Cascade Object Detection Face & Eye
import numpy as np
import cv2


# In[3]:


face_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




