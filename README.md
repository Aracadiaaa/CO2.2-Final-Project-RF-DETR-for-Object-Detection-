Load your Dataset by changing the
"
train_dataset = CustomDetectionDataset("Data/train/", "Data/train/annotations.json", get_transform())
val_dataset = CustomDetectionDataset("Data/valid/", "Data/valid/annotations.json", get_transform())
"
Be sure to make the Dataset in COCO Format.
Within the Training.py file and be sure to change the class names and change the number of classes it has within the Inference Module file
"
CLASS_NAMES = [
    "background",
    "bus-l-",
    "bus-s-",
    "car",
    "mid truck",
    "small bus",
    "small truck",
    "truck-l-",
    "truck-m-",
    "truck-s-",
    "truck-xl-"
]
"
+
"model = RFDETR(num_classes=12).to(DEVICE)   "
Change how many epochs you want to run and just run the Train.py module.
After it runs a new file will generate, it will be called the "rf_detr.pth". This file will be used for you RF-DETR for object detections.
