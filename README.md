Load your Dataset by changing the portion below in the training.py file. Be sure to make the Dataset in COCO Format.			
```
train_dataset = CustomDetectionDataset("Data/train/", "Data/train/annotations.json", get_transform())
val_dataset = CustomDetectionDataset("Data/valid/", "Data/valid/annotations.json", get_transform())
```
Be sure to change the class names and change the number of classes it has within the Inference Module file
```
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
```
```
"model = RFDETR(num_classes=12).to(DEVICE)
```
Change how many epochs you want to run and just run the Train.py module.
After it runs a new file will generate, it will be called the "rf_detr.pth". This file will be used for you RF-DETR for object detections.

Once you have successfully generated the rf_detr.ph, you must test the model. To do this, you must change the
```
image_path = "Data/test/adit_mp4-574_jpg.rf.2ce2d3744ed7299e8c2381451b63015a.jpg"   # <-- change this path to your test 
```
 to your own image file path and re-run the inference.py. This will open the COpenCV’s img display and show your sample image and a green box (label of the class). 

Thank you!
