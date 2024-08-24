from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

#Load the trained YOLOv8 classification model
model = YOLO('/home/alexander/runs/classify/train11/weights/best.pt')  # Replace with the path to your trained model

#Find 4 images you would like to test the model on. I am using the validation dataset but you can download them from Google if you want.
image_paths = ['/home/alexander/Test_data/val/kitten/000113.jpg', '/home/alexander/Test_data/val/car/000124.jpg', '/home/alexander/Test_data/val/puppy/000109.jpg', '/home/alexander/Test_data/val/motorbike/000065.jpg']

#Perform inference on each image
for image_path in image_paths:
    results = model.predict(image_path)
    
    #Display the image and prediction
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Convert to RGB for display

    #Get the predicted class label
    pred_class_index = results[0].probs.top1  # Extract class index with the highest probability
    pred_class = results[0].names[pred_class_index]  # Convert class index to class name

    #Get the confidence score for the top-1 prediction and format it as a percentage
    confidence = results[0].probs.top1conf  #Confidence score for the top-1 prediction
    confidence_percent = confidence.item() * 100  #Convert to percentage

    #Display the image with the predicted class label
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {pred_class} ({confidence_percent:.2f}%)")
    plt.axis('off')
    plt.show()



