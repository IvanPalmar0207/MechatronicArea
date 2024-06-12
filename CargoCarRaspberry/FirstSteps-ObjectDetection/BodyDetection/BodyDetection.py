import cv2

body = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

while True:
    
    _,frame = capture.read()
        
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodyColor = body.detectMultiScale(gray,1.2,5)
    
    for (x,y,h,w) in bodyColor:
        cv2.rectangle(frame,(x,y),(x + w, h + y), color = (51,204,204),thickness = 3)
        cv2.putText(frame,"Persona",(x,y), cv2.FONT_HERSHEY_COMPLEX, 1, color=(51,204,204), thickness= 2)
        
    cv2.imshow('WebCam',frame)
    
    if(cv2.waitKey(10) == ord('s')):
        break
    
capture.release()
cv2.destroyAllWindows()