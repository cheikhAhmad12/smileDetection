# Smile Detection

Binary smile / no-smile classifier with a Vision Transformer (ViT), offline image prediction, and webcam demos.

## Project layout
- `Data/SMILES/positives`, `Data/SMILES/negatives`: training images (grayscale is fine; will be converted to 3-channel).
- `train_vit.py`: end-to-end training/validation/test split, class weights, and metric/curve export to `result_vit/`.
- `models/`: pretrained assets (best ViT checkpoint, face detector Caffe files, legacy CNN).
- `predict_image.py`: load the saved ViT checkpoint and classify a single image.
- `webcam_smile_dnn.py`: live webcam smile detection using the OpenCV DNN face detector.
- `webcam_smile_haar.py`: alternative webcam demo using Haar cascades.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision timm scikit-learn pillow matplotlib tqdm opencv-python
```

## Train the ViT model
1) Put your dataset in `Data/SMILES/positives` and `Data/SMILES/negatives`.
2) Run:
```bash
python3 train_vit.py
```
Artifacts: `result_vit/best_vit_smile_detector.pth`, training curves, confusion matrix image, and report. Copy the best checkpoint to `models/best_vit_smile_detector.pth` if you want to use the inference scripts.

## Predict on an image
```bash
python3 predict_image.py 9157.jpg
# -> prints label (negatives/positives) and confidence
```
Make sure `models/best_vit_smile_detector.pth` exists (use the one from training results).

## Webcam demo (DNN face detector)
```bash
python3 webcam_smile_dnn.py
```
Requires `models/deploy.prototxt` and `models/res10_300x300_ssd_iter_140000.caffemodel` for face detection. Press `q` to quit. Use `webcam_smile_haar.py` if you prefer Haar cascades.

## Notes
- Training uses a 70/15/15 split with class weighting for imbalance.
- Transforms normalize to mean/std 0.5 and resize to 224x224; keep the same preprocessing for inference.
