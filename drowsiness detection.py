import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Khoi tao mixer de phat am thanh
mixer.init()
sound = mixer.Sound('alarm.wav')  # Nap file am thanh bao dong

# Nap file cascade de nhan dien khuon mat va mat
face = cv2.CascadeClassifier('haar cascade files\\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\\haarcascade_righteye_2splits.xml')

# Nhan lop phan loai (mo mat hoac nham mat)
lbl = ['Close', 'Open']

# Tai mo hinh CNN da duoc huan luyen
model = load_model('models/cnncat2.h5')

# Lay duong dan hien tai
path = os.getcwd()

# Mo webcam
cap = cv2.VideoCapture(0)

# Dinh nghia font chu
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Khoi tao cac bien
count = 0  # Dem so lan quet mat
score = 0  # Diem so phat hien tinh trang ngu gat
thicc = 2  # Do day cua vien bao dong
rpred = [99]  # Du doan cho mat phai
lpred = [99]  # Du doan cho mat trai

# Vong lap de xu ly tung frame
while(True):
    ret, frame = cap.read()  # Doc frame tu webcam
    height, width = frame.shape[:2]  # Lay chieu cao va rong cua frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyen frame sang anh xam

    # Phat hien khuon mat
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    # Phat hien mat trai va mat phai
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Ve hinh chu nhat o day duoi man hinh
    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Ve khung nhan dien khuon mat
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    # Phat hien tinh trang mat phai
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)  # Chuyen anh mat sang xam
        r_eye = cv2.resize(r_eye, (24, 24))  # Thay doi kich thuoc anh ve 24x24
        r_eye = r_eye / 255  # Chuan hoa pixel ve khoang [0, 1]
        r_eye = r_eye.reshape(24, 24, -1)  # Them mot truc vao cuoi
        r_eye = np.expand_dims(r_eye, axis=0)  # Mo rong them chieu de dua vao mo hinh
        rpred = model.predict_classes(r_eye)  # Du doan
        if (rpred[0] == 1):
            lbl = 'Open'  # Mat mo
        if (rpred[0] == 0):
            lbl = 'Closed'  # Mat nham
        break

    # Phat hien tinh trang mat trai
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)  # Chuyen anh mat sang xam
        l_eye = cv2.resize(l_eye, (24, 24))  # Thay doi kich thuoc anh ve 24x24
        l_eye = l_eye / 255  # Chuan hoa pixel ve khoang [0, 1]
        l_eye = l_eye.reshape(24, 24, -1)  # Them mot truc vao cuoi
        l_eye = np.expand_dims(l_eye, axis=0)  # Mo rong them chieu de dua vao mo hinh
        lpred = model.predict_classes(l_eye)  # Du doan
        if (lpred[0] == 1):
            lbl = 'Open'  # Mat mo
        if (lpred[0] == 0):
            lbl = 'Closed'  # Mat nham
        break

    # Neu ca hai mat deu nham
    if (rpred[0] == 0 and lpred[0] == 0):
        score = score + 1  # Tang diem so
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # Neu mot trong hai mat mo
    else:
        score = score - 1  # Giam diem so
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Giu diem so >= 0
    if (score < 0):
        score = 0
    # Hien thi diem so
    cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Neu diem so > 15, kich hoat bao dong
    if (score > 15):
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)  # Luu anh
        try:
            sound.play()  # Phat am thanh bao dong
        except:
            pass
        # Tang/giam do day vien bao dong
        if (thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        # Ve vien bao dong
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Hien thi frame len man hinh
    cv2.imshow('frame', frame)

    # Bam 'q' de thoat chuong trinh
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giai phong webcam va dong cua so
cap.release()
cv2.destroyAllWindows()
