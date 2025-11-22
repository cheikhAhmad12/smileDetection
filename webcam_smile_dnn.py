import cv2
import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
import os

# ======================
# CONFIG
# ======================
model_name = "vit_base_patch16_224"
model_path = "models/best_vit_smile_detector.pth"   # adapte si besoin
class_names = ["negatives", "positives"]                # 0 = pas sourire, 1 = sourire

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======================
# Transforms (mêmes que pour l'entraînement)
# ======================
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),
])

# ======================
# Charger le modèle sourire / non-sourire
# ======================
def load_smile_model():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable à : {model_path}")
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

smile_model = load_smile_model()

# ======================
# Charger le détecteur de visage DNN OpenCV
# ======================
proto_path = "models/deploy.prototxt"
model_face_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

if not (os.path.exists(proto_path) and os.path.exists(model_face_path)):
    raise FileNotFoundError("Fichiers du modèle de face DNN introuvables dans ./models")

face_net = cv2.dnn.readNetFromCaffe(proto_path, model_face_path)

# ======================
# Prédiction sur un crop de visage (BGR)
# ======================
def predict_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb).convert("L")

    x = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = smile_model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    return label, conf.item()

# ======================
# Boucle webcam
# ======================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    print("Appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossible de lire une frame de la webcam.")
            break

        (h, w) = frame.shape[:2]

        # Préparation pour le réseau DNN de détection de visage
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        face_net.setInput(blob)
        detections = face_net.forward()

        # Parcourir toutes les détections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Seuil de confiance pour accepter le visage
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_roi = frame[y1:y2, x1:x2]

            label, conf_smile = predict_face(face_roi)

            if label == "positives":
                text = f"Smile ({conf_smile:.2f})"
                color = (0, 255, 0)
            else:
                text = f"No smile ({conf_smile:.2f})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow("Smile detector (DNN face)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
