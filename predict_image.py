import torch
from torchvision import transforms
from PIL import Image
import timm
import os

# ======================
# CONFIG
# ======================
model_name = "vit_base_patch16_224"
model_path = "models/best_vit_smile_detector.pth"
class_names = ["negatives", "positives"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
#  Transforms identiques à l'entraînement
# ======================
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])


# ======================
#  Charger le modèle
# ======================
def load_model():
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ======================
#  Fonction de prédiction
# ======================
def predict_image(image_path):
    model = load_model()

    img = Image.open(image_path).convert("L")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    return label, confidence.item()


# ======================
#  Main
# ======================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 predict.py image.jpg")
        exit()

    image_path = sys.argv[1]

    label, conf = predict_image(image_path)
    print(f"Prediction: {label} (confidence: {conf:.2f})")
