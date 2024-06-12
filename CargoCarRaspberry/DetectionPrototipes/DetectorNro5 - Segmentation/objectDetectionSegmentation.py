from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.predict(source='./images/CarTest.jpg', save = True, conf = 0.5)