import cv2
import numpy as np


face_path = cv2.CascadeClassifier("C:/Users/adity/Desktop/Cascade/haarcascade_frontalface_default.xml")

def face_extractor(img):
    #img_umat = cv2.UMat(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_path.detectMultiScale(gray, 1.3, 5)
    
    if face in ():
        return None
        
    for(x, y, w, h) in face:
        cropped_face = img[y: y+h, x: x+w]
        
    return cropped_face
        
capture = cv2.VideoCapture(0)
count = 0

while True:
    
    ret, frame = capture.read()
    
    if face_extractor(frame) is not None:
        
        count = count + 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        #face_umat = cv2.UMat(face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
        output_path = "C:/Users/adity/Desktop/Cascade/Output_img/user"+str(count)+".jpg"
        cv2.imwrite(output_path, face)
                
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("face cropper", face)
            
    else:
        print("FACE NOT FOUND !")
        pass
        
    if cv2.waitKey(1) == 13 or count == 100:
        break
        
capture.release()
cv2.destroyAllWindows()
print("ALL SAMPLES COLLECTED !!")
        
        