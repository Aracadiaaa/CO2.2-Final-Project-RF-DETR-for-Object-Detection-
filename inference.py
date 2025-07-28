import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from model import RFDETR

# === Classes (12 + background) ===
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = RFDETR(num_classes=12).to(DEVICE)   # 12 actual classes + background = 13

model.load_state_dict(torch.load("rf_detr.pth", map_location=DEVICE))
model.eval()

# --- Image Transform ---
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor()
])

# === Load test image ===
image_path = "Data/test/adit_mp4-574_jpg.rf.2ce2d3744ed7299e8c2381451b63015a.jpg"   # <-- change this path to your test image
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# === Inference ===
with torch.no_grad():
    outputs = model(input_tensor)

pred_logits = outputs["pred_logits"][0]   # [num_queries, num_classes+1]
pred_boxes = outputs["pred_boxes"][0]     # [num_queries, 4] normalized

# --- Rescale boxes to original image size ---
w, h = image.size
pred_boxes = pred_boxes.cpu().numpy()
pred_boxes[:, 0] *= w
pred_boxes[:, 1] *= h
pred_boxes[:, 2] *= w
pred_boxes[:, 3] *= h

# --- Draw all predicted boxes ---
image_np = np.array(image)
for logit, box in zip(pred_logits, pred_boxes):
    score, label = torch.max(logit, dim=-1)
    label = label.item()
    score = score.item()

    if label == 0:  # skip background
        continue

    x1, y1, x2, y2 = box
    cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image_np, f"{CLASS_NAMES[label]} {score:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# === Show image popup ===
cv2.imshow("Detections", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
