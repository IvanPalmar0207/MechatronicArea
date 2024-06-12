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
    
#Classes to identify
print(classLabels)

#Classes len
print(len(classLabels))

#Reading an Image

image = cv2.imread('./Test/images/MotoGroup.jpg')

#BGR
'''plt.imshow(image) 
plt.show()'''

#Gray Scale
'''plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()'''

#Load the detection
classIndex, confidence, box = model.detect(image, confThreshold = 0.55 )

#The class index are the labels that the model is refering to, in this case are:
print(classIndex)
#Model's confidence for each object it detects
print(confidence)
#Axis of every object the model detects
print(box)

#Recognition of images
fontSize = 3
font = cv2.FONT_HERSHEY_COMPLEX
for classInd, confidece, boxes in zip(classIndex.flatten(),confidence.flatten(), box):
    cv2.rectangle(image,boxes,(238, 220, 35), 3)
    cv2.putText(image, classLabels[classInd-1], (boxes[0] + 10, boxes[1] + 40), font, fontScale = fontSize, color=(0, 225, 0), thickness = 3)
    
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show() 