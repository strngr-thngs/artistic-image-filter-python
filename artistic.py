import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

def spreadLookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

def make_warm(I):
  increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
  decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
  red, green, blue = cv2.split(I)
  red = cv2.LUT(red, decreaseLookupTable).astype(np.uint8)
  blue = cv2.LUT(blue, increaseLookupTable).astype(np.uint8)
  return cv2.merge((red, green, blue))

I = cv2.imread("./image.jpg")

G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
G = cv2.GaussianBlur(G, (3,3), 3.14)
G = cv2.medianBlur(G, 11)
G = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)

C = cv2.bilateralFilter(I, 7, 200, 200)

A = cv2.bitwise_and(C, C, mask=G)

E = cv2.Canny(C, 50, 200)
E[E==255] = 1
E[E==0] = 255
E[E==1] = 0
E = cv2.cvtColor(E, cv2.COLOR_GRAY2BGR)

alpha = 0.25
A = cv2.addWeighted(E, alpha, A, 1-alpha, 0)

alpha = 0.5
A = cv2.addWeighted(I, alpha, A, 1-alpha, 0)

A = make_warm(A)

cv2.imshow("I", I)
# cv2.imshow("", G)
# cv2.imshow("", C)
# cv2.imshow("", E)
cv2.imshow("A", A)
cv2.waitKey(0)
cv2.destroyAllWindows()
