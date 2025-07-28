import torch
from torch.utils.data import DataLoader
from dataset import CustomDetectionDataset, get_transform
from model import RFDETR
from matcher import HungarianMatcher
from losses import SetCriterion
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
train_dataset = CustomDetectionDataset("Data/train/", "Data/train/annotations.json", get_transform())
val_dataset = CustomDetectionDataset("Data/valid/", "Data/valid/annotations.json", get_transform())

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images)
    return images, list(targets)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Model & Loss
num_classes = 2  # change based on dataset
model = RFDETR(num_classes=12).to(device)
matcher = HungarianMatcher()
criterion = SetCriterion(num_classes=12, matcher=matcher, weight_dict={"loss_ce": 1, "loss_bbox": 5})
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def move_targets_to_device(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, targets in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        targets = move_targets_to_device(targets, device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = move_targets_to_device(targets, device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "rf_detr.pth")
print("Training complete and model saved as rf_detr.pth")
