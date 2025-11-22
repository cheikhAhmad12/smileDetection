import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import timm

# =====================
#  Disable HF progress bars (fix)
# =====================
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# =====================
#  Config de base
# =====================
print("restart")
data_dir = "Data/SMILES"
img_size = 224
batch_size = 32
num_workers = 4
num_epochs = 1

results_dir = "result_vit"
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# =====================
#  Transforms
# =====================
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])

# =====================
#  Dataset & split
# =====================
base_dataset = datasets.ImageFolder(root=data_dir, transform=None)
print("Classes ->", base_dataset.class_to_idx)
class_names = list(base_dataset.class_to_idx.keys())
num_classes = 2

total_len = len(base_dataset)
train_len = int(0.7 * total_len)
val_len   = int(0.15 * total_len)
test_len  = total_len - train_len - val_len

train_subset, val_subset, test_subset = random_split(
    base_dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42)
)

class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = img.convert("L")   # sécuriser la conversion en niveaux de gris
        if self.transform is not None:
            img = self.transform(img)
        return img, label

train_dataset = TransformDataset(train_subset, train_transforms)
val_dataset   = TransformDataset(val_subset,   val_transforms)
test_dataset  = TransformDataset(test_subset,  val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# =====================
#  Poids de classes
# =====================
n_neg = len(os.listdir(os.path.join(data_dir, "negatives")))
n_pos = len(os.listdir(os.path.join(data_dir, "positives")))

counts = np.array([n_neg, n_pos], dtype=float)
total  = counts.sum()
class_weights = total / (2.0 * counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights :", class_weights)

# =====================
#  Modèle ViT
# =====================
model_name = "vit_base_patch16_224"
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# =====================
#  Fonctions train/val
# =====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# =====================
#  Training + tracking
# =====================
best_val_acc = 0.0

train_losses = []
train_accs   = []
val_losses   = []
val_accs     = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, desc="Val")

    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Train - loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"Val   - loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(results_dir, "best_vit_smile_detector.pth"))
        print(">> New best model saved.")

print("\nTraining finished. Best val acc:", best_val_acc)

# =====================
#  Courbes
# =====================
epochs = range(1, num_epochs+1)

plt.figure()
plt.plot(epochs, train_losses, label="Train loss")
plt.plot(epochs, val_losses, label="Val loss")
plt.legend(); plt.grid(True)
plt.title("Loss curves")
plt.savefig(os.path.join(results_dir, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(epochs, train_accs, label="Train acc")
plt.plot(epochs, val_accs, label="Val acc")
plt.legend(); plt.grid(True)
plt.title("Accuracy curves")
plt.savefig(os.path.join(results_dir, "accuracy_curve.png"))
plt.close()

np.savez(os.path.join(results_dir, "training_history.npz"),
         train_losses=train_losses, val_losses=val_losses,
         train_accs=train_accs, val_accs=val_accs)

# =====================
#  Test evaluation
# =====================
best_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
best_model.load_state_dict(torch.load(os.path.join(results_dir, "best_vit_smile_detector.pth")))
best_model.to(device)

test_loss, test_acc = eval_one_epoch(best_model, test_loader, criterion, device, desc="Test")
print(f"\nTest - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

# =====================
#  Confusion matrix + classification report
# =====================
def get_all_preds(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)

y_true, y_pred = get_all_preds(best_model, test_loader)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", report)

# Save matrix
# Sauvegarde de la matrice de confusion en image
fig, ax = plt.subplots()

# on garde l'objet retourné par imshow
im = ax.imshow(cm, interpolation='nearest', cmap="Blues")

fig.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion matrix (test set)'
)

# Rotation des labels en x
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Afficher les valeurs dans chaque case
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix_test.png"), bbox_inches="tight")
plt.close()

