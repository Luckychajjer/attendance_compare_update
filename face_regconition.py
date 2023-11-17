import cv2
import numpy as np
import face_recognition
import os
import pandas as pd

def img_folder(path):
    images = []
    classNames = []
    myList = os.listdir(path)
    # print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    
    return images,classNames

def findEncodings(images):
    encodeList = []
    count =0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        count+=1
        try:
            encode = face_recognition.face_encodings(img)[0]
            print(count)
            encodeList.append(encode)
        except:
            pass
    return encodeList
 

def update_csv(csv_file, attendance_data):
    if not attendance_data:
        return
    
    df = pd.read_csv(csv_file)
    for name in attendance_data:
        df.loc[df['NAME'] == name, 'ATTENDANCE'] = "PRESENT"
    df.to_csv(csv_file, index=False)


def crop_photo_attendance():
    images_crop,classNames_crop=img_folder(path_crop)
    encodeListCrop = findEncodings(images_crop)
    print('Encoding Complete')
    present_student=[]
    for encodeFace in encodeListCrop:
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        print(matches)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            present_student.append(name)

    update_csv(path_csv,present_student)


def video_attendance(): 
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        #img = captureScreen()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        present_student=[]
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                present_student.append(name)

        update_csv(path_csv,present_student)
        cv2.imshow('Webcam',img)
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == 'q':
            break


if __name__ == "__main__":
    path = 'D:/PycharmProject/yoloface/main_database/family_db'
    path_crop = 'yoloface/cropped_images'
    path_csv = 'D:/PycharmProject/yoloface/attendence.CSV'
    images,classNames=img_folder(path)
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    crop_photo_attendance()
