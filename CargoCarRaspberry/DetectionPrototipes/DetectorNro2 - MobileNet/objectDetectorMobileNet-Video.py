#Packages
import cv2
import matplotlib.pyplot as plt

#Model and its config
configFile = './Models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozenModel = './Models/frozen_inference_graph.pb'

model = cv2.dnn.DetectionModel(frozenModel, configFile)

#Model configuration
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) #255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5)) #MobileNer => [-1,1]
model.setInputSwapRB(True)

#Labels
classLabels = []

fileName = './Models/classLabels.txt'

with open(fileName,'rt') as fn:
    classLabels = fn.read().rstrip('\n').split('\n')
    
#Video detection
capture = cv2.VideoCapture(0)

#The next code checks if the video y open correctly
if not capture.isOpened():
    capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise IOError('Could not open the video')

#Font Style
fontSize = 1
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    
    _, frame = capture.read()
    
    classIndex, confidence, box = model.detect(frame, confThreshold=0.55)
    
    print(classIndex, confidence, box)
    
    if(len(classIndex) != 0):
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(),box):
            if(classInd <= 80):
                cv2.rectangle(frame, boxes, (255,0,0), 2)
                cv2.putText(frame, classLabels[classInd-1],(boxes[0] + 10, boxes[1] + 40), font,fontSize,  color=(0, 225, 0), thickness = 2)
                
    cv2.imshow('Detection Object',frame)
        
    if(cv2.waitKey(2) & 0xFF == ord('s')):
        break
    
capture.release()
cv2.destroyAllWindows()