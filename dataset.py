import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import cv2
import json
import os
from PIL import Image

class CustomDetectionDataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms

        with open(ann_file) as f:
            coco_json = json.load(f)

        self.images = coco_json["images"]
        self.annotations = coco_json["annotations"]

        # Map image_id -> annotations
        self.imgid2anns = {}
        for ann in self.annotations:
            self.imgid2anns.setdefault(ann["image_id"], []).append(ann)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_folder, img_info["file_name"])

        # Read image with OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Annotations for this image
        annots = self.imgid2anns.get(img_info["id"], [])
        boxes = []
        labels = []
        for ann in annots:
            x, y, bw, bh = ann["bbox"]
            # Normalize bounding boxes (0-1)
            boxes.append([
                x / w,
                y / h,
                (x + bw) / w,
                (y + bh) / h
            ])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_info["id"]]),
        }

        # Convert to PIL for transforms
        img = Image.fromarray(img)
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

def get_transform():
    return T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
