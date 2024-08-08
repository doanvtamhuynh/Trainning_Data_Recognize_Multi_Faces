import cv2
import  numpy as np
from PIL import Image
import os

path = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/DataSet'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImageAndLables(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids

print('\n[INFO] Trainning...')

faces, ids = getImageAndLables(path)
recognizer.train(faces, np.array(ids))

pathTrain = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/Train/'
recognizer.write(pathTrain + 'trainer.yml')

print('\n[INFO] Complete')
