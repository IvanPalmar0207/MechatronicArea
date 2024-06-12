import cv2

#Arquitecture Model

prototxt = 'PreEntrenedModel/MobileNetSSD_deploy.prototxt.txt'

#Weights

model = 'PreEntrenedModel/MobileNetSSD_deploy.caffemodel'

#Class Labels

classes = {
	0 : 'background', 1 : 'aeroplane', 2 : 'bycicle',
	3 : 'bird', 4 : 'boat', 5 : 'bottle',
	6 : 'bus', 7 : 'car', 8 : 'cat',
	9 : 'chair', 10 : 'cow', 11 : 'diningtable',
	12 : 'dog', 13 : 'horse', 14 : 'motorbike',
	15 : 'person', 16 : 'pottedplant', 17 : 'sheep',
	18 : 'sofa', 19 : 'train', 20 : 'tvmonitor'
}

#Load Model

net = cv2.dnn.readNetFromCaffe(prototxt, model)


#Video Capture 

capture = cv2.VideoCapture('./Test/KoreaTest.mp4')

while True:
	
	ret ,frame = capture.read()
	
	if ret == False:
		break		
	
	height, width, _ = frame.shape
	frameResized = cv2.resize(frame, (300,300))
	
	#Creating a blob
	blob = cv2.dnn.blobFromImage(frameResized, 0.007843, (300,300), (127.5, 127.5, 127.5))
	
	#Detection and Prediction
	
	net.setInput(blob)
	detections = net.forward()
	
	for detection in detections[0][0]:		
		if detection[2] > 0.45:
			label = classes[detection[1]]
			print(f'Tipo de objeto: {label}\n')
			
			box = detection[3:7] * [width, height, width, height]
			xStart, yStart, xEnd, yEnd = int(box[0]), int(box[1]), int(box[2]), int(box[3])
			
			cv2.rectangle(frame, (xStart, yStart), (xEnd, yEnd), (0,255,0), 2)
			cv2.putText(frame,"conf: {:.2f}".format(detection[2] * 100), (xStart, yStart - 5), 1, 1.2, (255, 0, 0), 2)
			cv2.putText(frame, label, (xStart, yStart - 25), 1, 1.5, (0,255,0),2)
	
	cv2.imshow("Object's CApture",frame)
	if(cv2.waitKey(10) & 0xFF == 27):
		break

capture.release()
cv2.destroyAllWindows()
	








