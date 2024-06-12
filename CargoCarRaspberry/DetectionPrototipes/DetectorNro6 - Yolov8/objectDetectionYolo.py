'''
# Main Path #

1. Load yolov8 model
2. load video
3. load frames
4. detect objects
5. track objects
6. plot the results
7. visualize the results
'''

from ultralytics import YOLO
import cv2

#First Step
model = YOLO('yolov8n.pt')
print(f'Yolov8 model classes: {model.names}')

#Second Step
videoPath = './Videos/motoVideo.mp4'

capture = cv2.VideoCapture(videoPath)

ret = True
#Third Step

while ret:
    
    ret, frame = capture.read()
    
    if ret:
        #Fourth Step and Fifth Step
        results = model.track(frame,persist = True, classes = (0,1,2,3,4,5,6,7,8,9,10,11,13,16,17,18,24,26,37))
        
        #Plot the results        
        frameUnder = results[0].plot()
        
        #Viusalize
        cv2.imshow('WebCam Frames',frameUnder)
        if(cv2.waitKey(25) & 0xFF == ord('e')):
            break

capture.release()
cv2.destroyAllWindows()