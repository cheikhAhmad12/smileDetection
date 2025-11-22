import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np

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
# Charger le modèle
# ======================
def load_model():
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# ======================
# Détection de visage (Haar Cascade OpenCV)
# ======================
# On utilise le modèle frontalface fourni par OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Impossible de charger haarcascade_frontalface_default.xml")

# ======================
# Fonction de prédiction sur un crop de visage (numpy array BGR)
# ======================
def predict_face(face_bgr):
    # Convertir BGR (OpenCV) -> RGB -> PIL
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb).convert("L")  # niveau de gris

    x = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    return label, conf.item()

# ======================
# Boucle webcam
# ======================
def main():
    cap = cv2.VideoCapture(0)  # 0 = webcam par défaut

    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    print("Appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossible de lire une frame de la webcam.")
            break

        # frame est en BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection des visages
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]  # crop BGR

            label, conf = predict_face(face_roi)

            # Texte à afficher
            if label == "positives":
                text = f"Smile ({conf:.2f})"
                color = (0, 255, 0)  # vert
            else:
                text = f"No smile ({conf:.2f})"
                color = (0, 0, 255)  # rouge

            # Dessiner rectangle + texte
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("Smile detector", frame)

        # quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
