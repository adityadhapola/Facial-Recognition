import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = "C:/Users/adity/Desktop/Cascade/Output_img/" 
only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

train_data, labels = [],[] 

for i, files in enumerate(only_files):
    image_path = data_path + only_files[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    train_data.append(np.asarray(images, dtype= np.uint8))
    labels.append(i)
    
labels = np.asarray(labels, dtype = np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(train_data), np.asarray(labels))

print("Model trained succesfully")


face_path = cv2.CascadeClassifier("C:/Users/adity/Desktop/Cascade/haarcascade_frontalface_default.xml")


def face_dec(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_path.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return img, []
    
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + y, y + h), (0, 255, 0), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        
    return img, roi

capture = cv2.VideoCapture(0)
while True:
    
    ret, frame = capture.read()
    
    image, face = face_dec(frame)
    
    try:
        
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+ " % Confidence it is user"
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
        if confidence > 80:
            cv2.putText(image, "UNLOCKED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Cropper", image)
            
        else:
            cv2.putText(image, "LOCKED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Cropper", image)
            
    except:
            cv2.putText(image, "Face not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Cropper", image)
            pass
        
    if cv2.waitKey(1) == 13:
        break
    
capture.release()
cv2.destroyAllWindows() 
        