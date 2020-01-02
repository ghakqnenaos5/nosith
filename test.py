import cv2
import numpy as np
import matplotlib.pyplot as plt

def verticalProjection(img):
   "Return a list containing the sum of the pixels in each column"
   (h, w) = img.shape[:2]
   sumCols = []
   for j in range(w):
       col = img[0:h, j:j+1] # y1:y2, x1:x2
       sumCols.append(np.sum(col))
   return sumCols


im_gray = cv2.imread('/home/chris/001.JPG', cv2.IMREAD_GRAYSCALE)
binary = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('binary', binary)
# getting mask with connectComponents
ret, labels = cv2.connectedComponents(binary)
"""
for label in range(1,ret):
    mask = np.array(labels, dtype=np.uint8)
    mask[labels == label] = 255

    cv2.waitKey(0)
"""
proj = np.sum(binary, axis = 0)
print(proj)

plt.plot(proj)
plt.show()

"""
# getting ROIs with findContours
contours = cv2.findContours(projection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
for cnt in contours:
    (x,y,w,h) = cv2.boundingRect(cnt)
    ROI = projection[y:y+h,x:x+w]
    cv2.imshow('ROI', ROI)
    cv2.waitKey(0)
"""
cv2.destroyAllWindows()