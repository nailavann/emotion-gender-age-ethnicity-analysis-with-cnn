import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

emotion_labels = ['Sinirli','Igrenme','Korku','Mutlu','Uzgun', 'Saskin', 'Dogal']

etnik_labels = ['Beyaz',"Siyah","Asyalı",'Hintli',"Diğer"]

age_labels = ['Orta','Yasli','Genc']

gender_labels = ['Erkek','Kadin']

emotion_classifier =load_model('emotion_model.h5')
gender_classifier =load_model('gender_model.h5')
etnik_classifier = load_model('etnik_model.h5')
age_classifier = load_model('age_model.h5')

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


while cap.isOpened():
    _, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    faces = face_detector.detectMultiScale(gray,1.2,7)

   

    for (x,y,w,h) in faces:
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48))


        roi = roi_gray.astype('float32')/255.0
        roi = img_to_array(roi)   


        roi = np.expand_dims(roi,axis=0)
           
        

        prediction = emotion_classifier.predict(roi)[0]
        etnik_pred = etnik_classifier.predict(roi)[0]
        age_pred = age_classifier.predict(roi)[0]
        gender_pred = gender_classifier.predict(roi)[0]


       
        emotion=emotion_labels[prediction.argmax()]
        etnik = etnik_labels[etnik_pred.argmax()]
        age = age_labels[age_pred.argmax()]
        gender=gender_labels[gender_pred.argmax()]

        emotion_label = "{}: {:.2f}%".format(emotion, prediction[prediction.argmax()] * 100)
        etnik_label = "{}: {:.2f}%".format(etnik, etnik_pred[etnik_pred.argmax()] * 100)
        age_label = "{}: {:.2f}%".format(age, age_pred[age_pred.argmax()] * 100)
        gender_label = "{}: {:.2f}%".format(gender, gender_pred[gender_pred.argmax()] * 100)
        
        cv2.putText(frame,emotion_label,(x+45,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        cv2.putText(frame,gender_label,(x+45,y-35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        cv2.putText(frame,etnik_label,(x+45,y-65),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        cv2.putText(frame,age_label,(x+45,y-105),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

       
    cv2.imshow('image',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()