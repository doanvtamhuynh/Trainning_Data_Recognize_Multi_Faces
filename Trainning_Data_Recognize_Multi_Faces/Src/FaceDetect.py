import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,440)

face_defect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\nInsert Face ID ===>   ' )
print('\n[INFO] Camera Running...')
count = 0

while True:
    ret, img = cam.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_defect.detectMultiScale(gray, 1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        cv2.imwrite(path + "/DataSet/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 50:
        break

print('\n[INFO] Exit')
cam.release()
cv2.destroyAllWindows()