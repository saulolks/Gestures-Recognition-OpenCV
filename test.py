import os
import cv2
import imutils
import numpy as np


"""
    Abordagem 1:
    1. Detecta a regioes candidatas a ser a mao
    2. Validar as regioes encontradas
        2.1. Montar um conjunto de exemplos de objetos que sao mao e nao
    
    
    Abordagem 2:
    1. Detecta a regioes candidatas a ser a mao por meio de segmentação de imagem por intervalo de cores
    
    Abordagem 3:
    1. Detecta a regioes candidatas a ser a mao por meio do movimento 
     MOG2 KNN 
    
    ConvexHull
    
    Olhar algoritmo de similaridade e estrutural SSN
    
    3. Tracking dos objetos MOSSE, TKD, MEDIANSHIF
        3.1. Armazenar lista de imagens do tracking
    4. Classificar o gesto feito 


"""

# cap = cv2.VideoCapture('./videos/libras.mp4')
cap = cv2.VideoCapture('./videos/gesture.mp4')
ret = True

# cascade = cv2.CascadeClassifier('./haarcascades/hand.xml')
# cascade = cv2.CascadeClassifier('./haarcascades/haar-hand.xml')
cascade = cv2.CascadeClassifier('./haarcascades/palm_v4.xml')

SAVE_DIR = 'images'
countID = 0

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


while ret:

    ret, frame = cap.read()

    frame = imutils.resize(frame, 600)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite('{}/{:03d}.jpeg'.format(SAVE_DIR, countID), frame)
    cv2.imshow('img', frame)
    cv2.waitKey(1)
    countID +=1
    # cv2.destroyAllWindows()