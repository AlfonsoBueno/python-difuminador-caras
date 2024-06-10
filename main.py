# https://github.com/opencv/
# https://opencv.org/

import cv2 as cv

clasificador="haarcascade_frontalface_default.xml"
imagen="foto.jpg"

# Importamos fotografía a tratar
img = cv.imread(imagen)

gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_cascade=cv.CascadeClassifier(clasificador)

faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,minSize=(50,50),maxSize=(120,120))

for (x,y,w,h) in faces:
    face=img[y:y+h,x:x+w]
    blurred_face=cv.resize(cv.resize(face,(w//12,h//12)),(w,h))
    img[y:y+h,x:x+w]=blurred_face
   
    
cv.imshow("Ventana",img)
k=cv.waitKey(0)
cv.destroyAllWindows()