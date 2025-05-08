import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import pickle

# Emotion labels (in your training order)
emotions = ['angry','disgust', 'fear', 'happy' , 'neutral','sad','surprise']

# Load MobileNetV2 model
def load_mobilenet_v2(n_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.1),

        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.1),

        nn.Dropout(0.3),
        nn.Linear(256, n_classes)

    )
    return model

# Load and plot training history
try:
    with open("C:/Users/walam/PycharmProjects/facial expressions/efficientnet_b0_history.pkl", "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Could not load or plot history.pkl:", e)

# Load model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_mobilenet_v2(len(emotions)).to(device)
model=torch.load("C:/Users/walam/PycharmProjects/facial expressions/best_model_full_efficientnet_b0.pth", map_location=device)
model.eval()

# Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('./haarcascade_mcs_mouth.xml')

if face_cascade.empty() or mouth_cascade.empty():
    print("Error: Could not load Haar cascade files.")
    exit()

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Show emoji
def display_emoji(emotion_name):
    path = f'./emojis/{emotion_name}.png'
    emoji = cv2.imread(path)
    if emoji is not None:
        emoji_resized = cv2.resize(emoji, (100, 100))
        cv2.imshow('Emoji', emoji_resized)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_rgb = frame[y:y+h, x:x+w]

        # Emotion Prediction
        try:
            input_tensor = transform(face_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                emotion = emotions[pred]

                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2)
                display_emoji(emotion)
        except Exception as e:
            print("Error during emotion prediction:", e)

        # Detect mouth inside face region (lower half)
        roi_gray = gray[y + h//2:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=11)

        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(frame, (x + mx, y + h//2 + my), (x + mx + mw, y + h//2 + my + mh), (255, 0, 255), 2)
            break  # Only show the first detected mouth

    frame_resized = cv2.resize(frame, (700, 500))
    cv2.imshow('Facial Emotion Recognition', frame_resized)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

