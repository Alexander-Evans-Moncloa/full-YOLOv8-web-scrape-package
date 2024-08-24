from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')      #Loads pretrained model

#Trains model with only 1 epoch using my exercise images dataset. Change "alexander" to your own username. Remove device='CPU' if you have a good graphics card.

model.train(data='/home/alexander/Test_data', epochs = 20, imgsz = 64, device = 'CPU')